use axum::Json;
use tokio::net::{TcpListener, ToSocketAddrs};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_scalar::{Scalar, Servable};

pub(crate) mod audio;
mod error;

use crate::openai::audio::{AUDIO_DESC, AUDIO_TAG};
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
        (name = AUDIO_TAG, description = AUDIO_DESC)
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
    let handler = axum::serve(listener, router).await?;
    Ok(())
}

#[cfg(feature = "python")]
pub mod python {
    use crate::openai::audio::transcription::{
        TranscriptionRequest, TranscriptionResponse, TranscriptionRouter,
    };
    use crate::openai::serve_openai;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_core::{Endpoint, Handler};
    use pyo3::prelude::*;
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::Duration;
    use tokio::runtime::Builder;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use tokio::time::sleep;

    macro_rules! py_openai_endpoint_impl {
        ($name: ident, $router: ident, $request: ident, $response: ident) => {
            pub struct PyHandler {
                // Python allocated object with `Handler` protocol implementation
                inner: PyObject,
            }

            impl Handler for PyHandler {
                type Request = TranscriptionRequest;
                type Response = TranscriptionResponse;

                async fn on_request(&mut self, _request: Self::Request) -> Self::Response {
                    sleep(Duration::from_secs(2)).await;

                    TranscriptionResponse::Text(String::from("Coucou"))
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
                pub fn run(&self, interface: String, port: u16) -> PyResult<()> {
                    // Create the runtime
                    let rt = Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create runtime");

                    // IPC between the front running the API and the back executing the inference
                    let (sender, _receiver) =
                        unbounded_channel::<($request, UnboundedSender<$response>)>();

                    // Spawn the root task, scheduling all the underlying
                    rt.block_on(async move {
                        // let handler = Arc::clone(&self.handler);
                        if let Err(err) = serve_openai((interface, port), $router(sender)).await {
                            println!("Failed to start OpenAi compatible endpoint: {err}");
                        };
                    });

                    Ok(())
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
            .add_submodule(&crate::openai::audio::bind(py, &format!("{name}.audio"))?)?
            .finish();

        Ok(module)
    }
}
