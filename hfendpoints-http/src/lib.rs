use crate::headers::X_REQUEST_ID_NAME;
use std::fmt::Debug;
use tokio::net::{TcpListener, ToSocketAddrs};
use tower::ServiceBuilder;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::instrument;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_scalar::{Scalar, Servable};

mod api;
mod context;
pub mod environ;
pub mod error;
pub mod headers;
mod routes;

use crate::api::ApiDoc;
use crate::environ::Timeout;
use crate::routes::StatusRouter;
pub use context::Context;
pub use error::HttpError;
use hfendpoints_core::environ::TryFromEnv;
use hfendpoints_core::Error;

pub type HttpResult<T> = Result<T, HttpError>;
pub type RequestWithContext<I> = (I, Context);

const STATUS_TAG: &str = "Status";
const STATUS_DESC: &str = "Healthiness and monitoring of the endpoint";

pub const AUDIO_TAG: &str = "Audio";
pub const AUDIO_DESC: &str = "Learn how to turn audio into text or text into audio.";

pub const EMBEDDINGS_TAG: &str = "Embeddings";
pub const EMBEDDINGS_DESC: &str = "Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.";

#[instrument(skip(task_router))]
pub async fn serve_http<A, R>(interface: A, task_router: R) -> HttpResult<()>
where
    A: ToSocketAddrs + Debug,
    R: Into<OpenApiRouter>,
{
    // Retrieve the timeout duration from envvar
    let timeout = Timeout::try_from_env().map_err(|err| Error::Environment(err))?;

    // Default routes
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .merge(task_router.into())
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
                .layer(PropagateRequestIdLayer::new(X_REQUEST_ID_NAME.clone()))
                .layer::<TimeoutLayer>(timeout.into()),
        )
        .merge(StatusRouter::default().into())
        .split_for_parts();

    // Documentation route
    let router = router.merge(Scalar::with_url("/docs", api));

    let listener = TcpListener::bind(interface).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(feature = "python")]
pub mod python {
    use crate::Context;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;
    use pyo3_async_runtimes::TaskLocals;
    use tokio::sync::OnceCell;

    macro_rules! impl_pyhandler {
        ($handler: ident) => {
            use crate::python::TASK_LOCALS;
            use hfendpoints_core::{Error, Handler};
            use pyo3_async_runtimes::TaskLocals;
            use std::process;
            use tokio::sync::OnceCell;
            use tracing::{debug, info, instrument};

            pub(crate) static TASK_LOCALS: OnceCell<TaskLocals> = OnceCell::const_new();

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

    pub(crate) static TASK_LOCALS: OnceCell<TaskLocals> = OnceCell::const_new();

    pub async fn serve(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
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

    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .add_class::<Context>()?
            .finish();
        Ok(module)
    }
}
