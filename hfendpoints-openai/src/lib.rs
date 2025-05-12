use std::fmt::Debug;
use std::str::FromStr;
use tokio::net::ToSocketAddrs;

pub(crate) mod audio;
pub(crate) mod embeddings;

pub use hfendpoints_http::Context;

#[cfg(feature = "python")]
pub mod python {
    use hfendpoints_binding_python::tokio::create_multithreaded_runtime;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;
    use pyo3::prepare_freethreaded_python;
    use pyo3_async_runtimes::tokio::init;
    use pyo3_async_runtimes::TaskLocals;
    use tokio::sync::OnceCell;
    use tracing::instrument;

    pub(crate) static TASK_LOCALS: OnceCell<TaskLocals> = OnceCell::const_new();

    macro_rules! impl_pyhandler {
        ($handler: ident) => {
            use crate::python::TASK_LOCALS;
            use hfendpoints_core::{Error, Handler};
            use pyo3_async_runtimes::TaskLocals;
            use std::process;
            use tokio::sync::OnceCell;
            use tracing::{debug, info, instrument};

            /// Wraps the underlying, Python's heap-allocated, object in a GIL independent way
            /// to be shared between threads.
            ///
            /// `PyHandler` implements the `Handler` trait which forwards the request handling
            /// logic back to Python through the `hfendpoints.Handler` protocol enforcing
            /// implementation of `__call__` method.
            ///
            pub struct PyHandler {
                /// Python allocated object with `Handler` protocol implementation, GIL-independent
                inner: PyObject,
            }

            impl $handler for PyHandler {
                #[instrument(skip_all)]
                async fn on_request(
                    &self,
                    request: Self::Request,
                ) -> Result<Self::Response, Error> {
                    // Retrieve the current event loop
                    let locals = Python::with_gil(|py| TASK_LOCALS.get().unwrap().clone_ref(py));

                    let (request, ctx) = request;

                    // Create the coroutine on Python side to await through tokio
                    let coro = Python::with_gil(|py| {
                        let py_coro_call = self.inner.call1(py, (request, ctx))?.into_bound(py);

                        debug!("[NATIVE] asyncio Handler's coroutine (__call__) created");
                        pyo3_async_runtimes::into_future_with_locals(&locals, py_coro_call)
                    })
                    .inspect_err(|err| {
                        error!("Failed to retrieve __call__ coroutine: {err}");
                        Err(Error::Handler(Implementation("Handler is not callable.")))
                    })?;

                    pyo3_async_runtimes::tokio::get_runtime()
                        .spawn(async {
                            // Schedule the coroutine
                            let response = pyo3_async_runtimes::tokio::scope(locals, coro)
                                .await
                                .inspect_err(|err| {
                                    error!("Failed to execute __call__: {err}");
                                })?;

                            debug!("[NATIVE] asyncio Handler's coroutine (__call__) done");

                            // We are downcasting from Python object to Rust typed type
                            Ok(Python::with_gil(|py| {
                                response.extract::<Self::Response>(py)
                            })?)
                        })
                        .await
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
            }
        };
    }

    macro_rules! impl_pyendpoint {
        ($name: literal, $pyname: ident, $handler: ident, $router: ident) => {
            use crate::{__path_health, ApiDoc, Context, health, serve_http};
            use hfendpoints_core::{Endpoint, wait_for_requests};
            use pyo3::exceptions::PyRuntimeError;
            use pyo3::prelude::*;
            use pyo3::types::PyNone;
            use std::sync::Arc;
            use tokio::net::TcpListener;
            use tokio::sync::mpsc::unbounded_channel;
            use tokio::task::spawn;
            use tracing::error;
            use utoipa::OpenApi;
            use utoipa_axum::{router::OpenApiRouter, routes};
            use utoipa_scalar::{Scalar, Servable};

            #[pyclass(name = $name)]
            pub(crate) struct $pyname(Arc<$handler>);

            impl Endpoint<(String, u16)> for $pyname {
                #[instrument(skip_all)]
                async fn serve(&self, inet_address: (String, u16)) -> Result<(), Error> {
                    let (sender, receiver) = unbounded_channel();
                    let router = $router { 0: sender };

                    // Handler in another thread
                    let handler = Arc::clone(&self.0);
                    let _ = pyo3_async_runtimes::tokio::get_runtime()
                        .spawn(wait_for_requests(receiver, handler));

                    info!(
                        "Starting endpoint at {}:{}",
                        &inet_address.0, &inet_address.1
                    );
                    pyo3_async_runtimes::tokio::get_runtime()
                        .spawn(serve_http(inet_address, router))
                        .await
                        .inspect_err(|err| {
                            info!("Caught error while serving endpoint: {err}");
                        })
                        .unwrap();

                    Ok(())
                }
            }

            #[pymethods]
            impl $pyname {
                #[instrument(skip(inner))]
                #[new]
                fn new(inner: PyObject) -> Self {
                    Self {
                        0: Arc::new(PyHandler { inner }),
                    }
                }

                #[instrument(skip_all)]
                async fn _serve_(&self, interface: String, port: u16) -> PyResult<()> {
                    if let Err(err) = self.serve((interface, port)).await {
                        error!("Caught error while serving HTTP endpoint: {err}");
                        Err(PyRuntimeError::new_err(err.to_string()))
                    } else {
                        Ok(())
                    }
                }
            }
        };
    }

    async fn serve(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
        let locals = TASK_LOCALS
            .get_or_try_init(|| async {
                Python::with_gil(|py| pyo3_async_runtimes::tokio::get_current_locals(py))
            })
            .await?;

        Python::with_gil(|py| {
            let coro = endpoint
                .bind(py)
                .call_method1("_serve_", (interface, port))?;
            pyo3_async_runtimes::into_future_with_locals(&locals, coro)
        })?
        .await?;
        Ok(())
    }

    #[pyfunction]
    #[instrument(skip(endpoint))]
    #[pyo3(name = "run")]
    fn run(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
        prepare_freethreaded_python();

        // Initialize the tokio runtime and bind this runtime to the tokio <> asyncio compatible layer
        init(create_multithreaded_runtime());

        Python::with_gil(|py| {
            py.allow_threads(|| {
                pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                    Python::with_gil(|inner| {
                        pyo3_async_runtimes::tokio::run(inner, serve(endpoint, interface, port))
                    })?;
                    Ok::<_, PyErr>(())
                })
            })
        })?;

        Ok::<_, PyErr>(())
    }

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .finish();

        module.add_function(wrap_pyfunction!(run, &module)?)?;
        Ok(module)
    }
}
