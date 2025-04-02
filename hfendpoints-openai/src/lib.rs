use axum::Json;
use tokio::net::{TcpListener, ToSocketAddrs};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

pub(crate) mod audio;
mod error;

use crate::audio::{AUDIO_DESC, AUDIO_TAG};
use error::OpenAiError;

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
    use crate::audio::transcription::{
        TranscriptionRequest, TranscriptionResponse, TranscriptionRouter,
    };
    use crate::serve_openai;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_core::{spawn_handler, Endpoint, Error, Handler};
    use pyo3::prelude::*;
    use pyo3::types::PyNone;
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::Duration;
    use tokio::runtime::Builder;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use tracing::{error, info};

    macro_rules! py_openai_endpoint_impl {
        ($name: ident, $router: ident, $request: ident, $response: ident) => {
            pub struct PyHandler {
                // Python allocated object with `Handler` protocol implementation
                inner: PyObject,
            }

            impl Handler for PyHandler {
                type Request = TranscriptionRequest;
                type Response = TranscriptionResponse;

                fn on_request(&self, request: Self::Request) -> Result<Self::Response, Error> {
                    info!("[FFI] Calling Python Handler");

                    Ok(Python::with_gil(|py| {
                        self.inner
                            .call(py, (request, PyNone::get(py)), None)?
                            .extract::<TranscriptionResponse>(py)
                    })?)
                }
            }

            #[pyclass]
            pub struct $name {
                handler: Arc<PyHandler>,
            }

            impl Endpoint for $name {
                fn spawn_handler(&self) -> JoinHandle<()> {
                    std::thread::spawn(|| {
                        std::thread::sleep(Duration::from_secs(10));
                    })
                }
            }

            #[pymethods]
            impl $name {
                #[new]
                #[pyo3(signature = (handler,))]
                pub fn new(handler: PyObject) -> PyResult<Self> {
                    Ok(Self {
                        handler: Arc::new(PyHandler { inner: handler }),
                    })
                }

                #[pyo3(signature = (interface, port))]
                pub fn run(&self, py: Python<'_>, interface: String, port: u16) -> PyResult<()> {
                    py.allow_threads(|| {
                        // Create the runtime
                        let rt = Builder::new_multi_thread()
                            .enable_all()
                            .build()
                            .expect("Failed to create runtime");

                        // IPC between the front running the API and the back executing the inference
                        let background_handler = Arc::clone(&self.handler);
                        let (sender, receiver) = unbounded_channel::<(
                            $request,
                            UnboundedSender<Result<$response, Error>>,
                        )>();

                        info!("[LOOPER] Spawning inference thread");
                        let inference_handle = spawn_handler(receiver, background_handler);

                        // Spawn the root task, scheduling all the underlying
                        rt.block_on(async move {
                            if let Err(err) = serve_openai((interface, port), $router(sender)).await
                            {
                                error!("Failed to start OpenAi compatible endpoint: {err}");
                            };
                        });

                        let _ = inference_handle.join();
                        Ok(())
                    })
                }
            }
        };
    }

    py_openai_endpoint_impl!(
        AutomaticSpeechRecognitionEndpoint,
        TranscriptionRouter,
        TranscriptionRequest,
        TranscriptionResponse
    );

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_submodule(&crate::audio::bind(py, &format!("{name}.audio"))?)?
            .finish();

        Ok(module)
    }
}
