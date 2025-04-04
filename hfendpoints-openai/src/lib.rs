use crate::audio::{AUDIO_DESC, AUDIO_TAG};
use axum::Json;
use error::OpenAiError;
use tokio::net::{TcpListener, ToSocketAddrs};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

pub(crate) mod audio;
mod error;

type OpenAiResult<T> = Result<T, OpenAiError>;

const STATUS_TAG: &str = "Status";
const STATUS_DESC: &str = "Healthiness and monitoring of the endpoint";

#[utoipa::path(
    method(get, head),
    path = "/health",
    tag = STATUS_TAG,
    responses(
        (status = OK, description = "Success", body = str, content_type = "application/json")
    )
)]
async fn health() -> Json<&'static str> {
    Json::from("OK")
}

#[derive(OpenApi)]
#[openapi(
    info(title = "Hugging Face Inference Endpoint Open AI Compatible Endpoint"),
    tags(
        (name = STATUS_TAG, description = STATUS_DESC),
        (name = AUDIO_TAG, description = AUDIO_DESC),
    )
)]
struct ApiDoc;

pub async fn serve_openai<A, R>(interface: A, task_router: R) -> OpenAiResult<()>
where
    A: ToSocketAddrs,
    R: Into<OpenApiRouter>,
{
    // Default routes
    let router = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(health))
        .nest("/api/v1", task_router.into());

    let (router, api) = router.split_for_parts();

    // Documentation route
    let router = router.merge(Scalar::with_url("/docs", api));

    let listener = TcpListener::bind(interface).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

pub trait EndpointRouter {
    type Request;
    type Response;
}

#[cfg(feature = "python")]
pub mod python {
    use hfendpoints_binding_python::tokio::create_multithreaded_runtime;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_core::Endpoint;
    use pyo3::prelude::*;
    use pyo3::prepare_freethreaded_python;
    use pyo3_async_runtimes::tokio::init;
    use std::sync::Arc;

    macro_rules! impl_pyhandler {
        ($request: ident, $response: ident) => {
            use hfendpoints_core::{Error, Handler};
            use tracing::{debug, info};
            use tokio::sync::OnceCell;
            use pyo3_async_runtimes::TaskLocals;

            static TASK_LOCALS: OnceCell<TaskLocals> = OnceCell::const_new();


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

            impl Handler for PyHandler {
                type Request = $request;
                type Response = $response;

                async fn on_request(&self, request: Self::Request) -> Result<Self::Response, Error> {
                    info!("[FFI] Calling Python Handler");
                    let _ = TASK_LOCALS.get_or_try_init(|| async {
                        Python::with_gil(|py| {
                            pyo3_async_runtimes::tokio::get_current_locals(py)
                        })
                    }).await?;

                    let response = Python::with_gil(|py| {
                        debug!("[NATIVE] Python's GIL acquired");

                        let py_coro_call = self.inner
                            .bind(py)
                            .call((request, PyNone::get(py)), None)?;



                        let coro = pyo3_async_runtimes::tokio::into_future(py_coro_call);
                        debug!("[NATIVE] asyncio Handler's coroutine (__call__) created");

                        coro
                    })?;

                    // We are delegating the future polling back to Rust
                    let response = response.await?;

                    debug!("[NATIVE] asyncio Handler's coroutine (__call__) done");

                    // We are downcasting from Python object to Rust typed type
                    Ok(Python::with_gil(|py| response.extract::<Self::Response>(py))?)
                }
            }

            // #[pyclass]
            // pub struct $name {
            //     handler: Arc<PyHandler>,
            // }
            //
            // #[pymethods]
            // impl $name {
            //     #[new]
            //     #[pyo3(signature = (handler,))]
            //     pub fn new(handler: PyObject) -> PyResult<Self> {
            //         Ok(Self {
            //             handler: Arc::new(PyHandler { inner: handler }),
            //         })
            //     }

                // #[pyo3(signature = (interface, port))]
                // pub fn run(&self, py: Python<'_>, interface: String, port: u16) -> PyResult<()> {
                //     py.allow_threads(|| {
                //         // Global tokio runtime behind Python as main thread
                //         debug!("Initializing tokio runtime from Python");
                //         init_tokio_runtime(create_multithreaded_runtime());
                //         let runtime = get_runtime();
                //
                //         // IPC between the front running the API and the back executing the inference
                //         let background_handler = Arc::clone(&self.handler);
                //         let (sender, receiver) = unbounded_channel::<(
                //             $request,
                //             UnboundedSender<Result<$response, Error>>,
                //         )>();
                //
                //         info!("[LOOPER] Spawning inference thread");
                //         let handler_loop_handle = runtime.spawn(
                //             wait_for_requests(receiver, background_handler)
                //         );
                //
                //         info!("[LOOPER] Spawned inference thread");
                //
                //         // Spawn the root task, scheduling all the underlying
                //         runtime.block_on(async move {
                //             if let Err(err) = try_join!(serve_openai((interface, port), $router(sender)))
                //             {
                //                 error!("Failed to start OpenAi compatible endpoint: {err}");
                //             };
                //         });
                //
                //         // Wait for the inference thread to finish and cleanup
                //         let _ = runtime.block_on(handler_loop_handle);
                //
                //         Ok(())
                //     })
                // }
            // }
        };
    }

    macro_rules! impl_pyendpoint {
        ($name: literal, $pyname: ident, $handler: ident, $router: ident) => {
            use crate::{ApiDoc, health, __path_health, serve_openai};
            use hfendpoints_core::{Endpoint, wait_for_requests};
            use std::sync::Arc;
            use pyo3::exceptions::PyRuntimeError;
            use pyo3::types::PyNone;
            use pyo3::prelude::*;
            use tokio::net::TcpListener;
            use tokio::sync::mpsc::unbounded_channel;
            use tokio::task::spawn;
            use tracing::error;
            use utoipa::OpenApi;
            use utoipa_axum::{routes, router::OpenApiRouter};
            use utoipa_scalar::{Scalar, Servable};

            #[pyclass(name = $name)]
            pub(crate) struct $pyname(Arc<$handler>);

            impl Endpoint<(String, u16)> for $pyname {
                async fn serve(&self, inet_address: (String, u16)) -> Result<(), Error> {
                    let (sender, receiver) = unbounded_channel();
                    let router = $router { 0: sender };

                    // Handler in another thread
                    let handler = Arc::clone(&self.0);
                    let _ = spawn(wait_for_requests(receiver, handler));

                    info!("Starting OpenAi compatible endpoint at {}:{}", &inet_address.0, &inet_address.1);
                    serve_openai(inet_address, router).await.inspect_err(|err|{
                        info!("Caught error while serving endpoint: {err}");
                    }).unwrap();

                    Ok(())
                }
            }

            #[pymethods]
            impl $pyname {
                #[new]
                fn new(inner: PyObject) -> Self {
                    Self { 0: Arc::new(PyHandler { inner }) }
                }

                async fn _serve_(&self, interface: String, port: u16) -> PyResult<()> {
                    if let Err(err) = self.serve((interface, port)).await {
                        error!("Caught error while serving Open Ai compatible endpoint: {err}");
                        Err(PyRuntimeError::new_err(err.to_string()))
                    } else {
                        Ok(())
                    }
                }
            }
        };
    }

    pub(crate) use impl_pyendpoint;
    pub(crate) use impl_pyhandler;


    async fn serve(endpoint: Arc<PyObject>, interface: String, port: u16) -> PyResult<()> {
        Python::with_gil(|py| {
            let coro = endpoint.bind(py).call_method1("_serve_", (interface, port))?;
            pyo3_async_runtimes::tokio::into_future(coro)
        })?.await?;
        Ok(())
    }

    #[pyfunction]
    #[pyo3(name = "run")]
    fn run(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
        prepare_freethreaded_python();

        // Initialize the tokio runtime and bind this runtime to the tokio <> asyncio compatible layer
        init(create_multithreaded_runtime());

        let endpoint = Arc::new(endpoint);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let endpoint = Arc::clone(&endpoint);
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
            .add_submodule(&crate::audio::python::bind(py, &format!("{name}.audio"))?)?
            .finish();

        module.add_function(wrap_pyfunction!(run, &module)?)?;
        Ok(module)
    }
}
