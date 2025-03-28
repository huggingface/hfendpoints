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

pub async fn serve_openai<A: ToSocketAddrs>(interface: A) -> OpenAiResult<()> {
    let listener = TcpListener::bind(interface).await?;
    let router = OpenApiRouter::with_openapi(ApiDoc::openapi()).routes(routes!(health));

    let (router, api) = router.split_for_parts();
    let router = router.merge(Scalar::with_url("/docs", api));

    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(feature = "python")]
pub mod python {
    use crate::openai::serve_openai;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;
    use tokio::runtime::Builder;

    macro_rules! py_openai_endpoint_impl {
        ($pyname: ident, $name: literal) => {
            #[pyclass(name = $name, subclass)]
            pub struct $pyname {}

            #[pymethods]
            impl $pyname {
                #[new]
                pub fn new() -> PyResult<Self> {
                    Ok(Self {})
                }

                #[pyo3(signature = (interface, port))]
                pub fn run(&self, interface: String, port: u16) -> PyResult<()> {
                    // Create the runtime
                    let rt = Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("Failed to create runtime");

                    // Spawn the root task, scheduling all the underlying
                    rt.block_on(async move {
                        if let Err(err) = serve_openai((interface, port)).await {
                            println!("Failed to start OpenAi compatible endpoint: {err}");
                        };
                    });

                    Ok(())
                }
            }
        };
    }

    py_openai_endpoint_impl!(
        PyAutomaticSpeechRecognitionEndpoint,
        "AutomaticSpeechRecognitionEndpoint"
    );

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<PyAutomaticSpeechRecognitionEndpoint>()?
            .finish();

        Ok(module)
    }
}
