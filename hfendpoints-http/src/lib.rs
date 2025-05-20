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

pub(crate) mod api;
mod context;
pub mod environ;
pub mod error;
pub mod headers;
pub mod routes;

pub use crate::api::ApiDoc;
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
    let timeout = Timeout::try_from_env().map_err(Error::Environment)?;

    // Default routes
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .merge(task_router.into())
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
                .layer(PropagateRequestIdLayer::new(X_REQUEST_ID_NAME.clone()))
                .layer(TimeoutLayer::from(timeout)),
        )
        .merge(StatusRouter.into())
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
    use hfendpoints_binding_python::tokio::create_multithreaded_runtime;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;
    use pyo3::prepare_freethreaded_python;
    use pyo3_async_runtimes::tokio::init;
    use pyo3_async_runtimes::TaskLocals;
    use tokio::sync::OnceCell;
    use tracing::instrument;

    #[macro_export]
    macro_rules! impl_http_pyhandler {
        ($request: ident, $response: ident, $pyrequest: ident, $pyresponse: ident) => {
            use hfendpoints_core::{EndpointResult, Error, Handler, HandlerError::Implementation};
            use pyo3::exceptions::PyRuntimeError;
            use pyo3::prelude::*;
            use pyo3_async_runtimes::TaskLocals;
            use std::process;
            use tokio::sync::OnceCell;
            use tracing::{debug, error, info, instrument};
            use $crate::Context;
            use $crate::python::TASK_LOCALS;

            #[pyclass(subclass)]
            pub struct PyHandler {
                inner: PyObject,
            }

            impl PyHandler {
                fn materialize_coroutine(
                    &self,
                    request: $pyrequest,
                    context: Context,
                    locals: &TaskLocals,
                ) -> EndpointResult<impl Future<Output = PyResult<PyObject>> + Send + 'static> {
                    Python::with_gil(|py| {
                        // Only pass the request part to Python
                        let py_coro_call = self.inner.call1(py, (request, context))?.into_bound(py);

                        debug!("[NATIVE] asyncio Handler's coroutine (__call__) created");
                        pyo3_async_runtimes::into_future_with_locals(&locals, py_coro_call)
                    })
                    .map_err(|err| {
                        error!("Failed to retrieve __call__ coroutine: {err}");
                        Error::from(Implementation(err.to_string().into()))
                    })
                }

                async fn execute_coroutine(
                    &self,
                    coroutine: impl Future<Output = PyResult<PyObject>> + Send + 'static,
                    locals: TaskLocals,
                ) -> PyResult<$pyresponse> {
                    pyo3_async_runtimes::tokio::get_runtime()
                        .spawn(async {
                            // Schedule the coroutine
                            let response =
                                match pyo3_async_runtimes::tokio::scope(locals, coroutine).await {
                                    Ok(resp) => resp,
                                    Err(err) => {
                                        error!("Failed to execute __call__: {err}");
                                        return Err(err);
                                    }
                                };

                            debug!("[NATIVE] asyncio Handler's coroutine (__call__) done");

                            // We are downcasting from a Python object to a Rust typed object
                            match Python::with_gil(|py| response.extract::<$pyresponse>(py)) {
                                Ok(resp) => Ok(resp),
                                Err(err) => Err(err),
                            }
                        })
                        .await
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
            }

            impl Handler for PyHandler {
                type Request = ($request, Context);
                type Response = $response;

                async fn on_request(
                    &self,
                    request: Self::Request,
                ) -> Result<Self::Response, Error> {
                    // Retrieve the current event loop
                    let locals = Python::with_gil(|py| TASK_LOCALS.get().unwrap().clone_ref(py));
                    let (request, context) = request;

                    debug!("[INGRESS] request: {request:?}");

                    // Convert the underlying frontend-specific message to the I/O adapter layer
                    let request = $pyrequest(request.try_into()?);

                    debug!("[INGRESS] successfully converted request");

                    // Create the coroutine on the Python side to await through tokio
                    let coroutine = self.materialize_coroutine(request, context, &locals)?;

                    // Execute the coroutine on Python
                    let response = self.execute_coroutine(coroutine, locals).await?;

                    // Attempt to convert back the output to the original frontend-specific message
                    response.0.try_into()
                }
            }
        };
    }

    #[macro_export]
    macro_rules! impl_http_pyendpoint {
        ($name: literal, $pyname: ident, $handler: ident, $router: ident) => {
            use hfendpoints_core::{Endpoint, wait_for_requests};
            use pyo3::prelude::*;
            use pyo3::types::PyNone;
            use std::sync::Arc;
            use tokio::net::TcpListener;
            use tokio::sync::mpsc::unbounded_channel;
            use tokio::task::spawn;
            use utoipa::OpenApi;
            use utoipa_axum::{router::OpenApiRouter, routes};
            use $crate::routes::{__path_health, health};
            use $crate::{ApiDoc, serve_http};

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

                    match pyo3_async_runtimes::tokio::get_runtime()
                        .spawn(serve_http(inet_address, router))
                        .await
                    {
                        Ok(res) => Ok(res?),
                        Err(join_error) => Err(Error::Runtime(join_error.to_string().into())),
                    }
                }
            }

            #[pymethods]
            impl $pyname {
                #[instrument(skip(inner))]
                #[new]
                fn new(inner: PyObject) -> Self {
                    Self(Arc::new($handler { inner }))
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

    // `TaskLocal` keeps track of the asynchronous computations being run globally for the whole tokio-spawned threads
    pub static TASK_LOCALS: OnceCell<TaskLocals> = OnceCell::const_new();

    pub async fn serve(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
        let locals = TASK_LOCALS
            .get_or_try_init(|| async {
                Python::with_gil(pyo3_async_runtimes::tokio::get_current_locals)
            })
            .await?;

        Python::with_gil(|py| {
            let coro = endpoint
                .bind(py)
                .call_method1("_serve_", (interface, port))?;
            pyo3_async_runtimes::into_future_with_locals(locals, coro)
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

    /// Bind this module to the python's wheel  
    ///
    /// # Arguments
    ///
    /// * `py`: Python acquired GIL reference
    /// * `name`: name of the python module to register this under
    ///
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<Context>()?
            .finish();

        module.add_function(wrap_pyfunction!(run, &module)?)?;

        Ok(module)
    }
}
