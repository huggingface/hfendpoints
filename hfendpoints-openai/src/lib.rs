use crate::audio::{AUDIO_DESC, AUDIO_TAG};
use axum::http::{HeaderName, StatusCode};
use error::OpenAiError;
use std::fmt::Debug;
use tokio::net::{TcpListener, ToSocketAddrs};
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;
use tracing::instrument;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

pub(crate) mod audio;
mod context;
pub(crate) mod embeddings;
mod error;
mod headers;

pub use context::Context;

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
#[instrument]
async fn health() -> StatusCode {
    StatusCode::OK
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

#[instrument(skip(task_router))]
pub async fn serve_openai<A, R>(interface: A, task_router: R) -> OpenAiResult<()>
where
    A: ToSocketAddrs + Debug,
    R: Into<OpenApiRouter>,
{
    // Correlation-ID middleware (x-request-id)
    let x_request_id_header_name = HeaderName::from_static("x-request-id");

    // Default routes
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .nest("/api/v1", task_router.into())
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
                .layer(PropagateRequestIdLayer::new(x_request_id_header_name)),
        )
        .routes(routes!(health))
        .split_for_parts();

    // Documentation route
    let router = router.merge(Scalar::with_url("/docs", api));

    let listener = TcpListener::bind(interface).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

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
        ($request: ident, $response: ident) => {
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

            impl Handler for PyHandler {
                type Request = ($request, Context);
                type Response = $response;

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
            use crate::{__path_health, ApiDoc, Context, health, serve_openai};
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
                #[instrument(skip(self))]
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
                        .spawn(serve_openai(inet_address, router))
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

                #[instrument(skip(self))]
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

    use crate::context::Context;
    pub(crate) use impl_pyendpoint;
    pub(crate) use impl_pyhandler;

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
            .add_class::<Context>()?
            .add_submodule(&crate::audio::python::bind(py, &format!("{name}.audio"))?)?
            .add_submodule(&crate::embeddings::python::bind(
                py,
                &format!("{name}.embeddings"),
            )?)?
            .finish();

        module.add_function(wrap_pyfunction!(run, &module)?)?;
        Ok(module)
    }
}
