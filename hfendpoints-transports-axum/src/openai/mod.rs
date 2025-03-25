use axum::{Json, Router};
use std::ops::Deref;
use thiserror::Error;
use tokio::io::Error as TokioIoError;
use tokio::net::{TcpListener, ToSocketAddrs};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_redoc::{Redoc, Servable};

mod audio;

use audio::{AUDIO_DESC, AUDIO_TAG};

/// Define all the possible errors for OpenAI Compatible Endpoint
#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("I/O Error occured: {0}")]
    IoError(#[from] TokioIoError),
}

/// Interface describing how to create an OpenAiRouter to be consumed by
/// the `OpenAiEndpoint` for allocating and exposing services to the network.
pub(crate) trait OpenAiRouterFactory {
    fn description() -> &'static str;
    fn routes() -> OpenApiRouter;
}

#[derive(OpenApi)]
#[openapi(
    info(title = "Hugging Face Inference Endpoint Open AI Compatible Endpoint"),
    tags(
        (name = AUDIO_TAG, description = AUDIO_DESC)
    )
)]
struct ApiDoc;

#[utoipa::path(
    method(get, head),
    path = "/health",
    responses(
        (status = OK, description = "Success", body = str, content_type = "application/json")
    )
)]
async fn health() -> Json<&'static str> {
    Json::from("OK")
}

///
pub struct OpenAiEndpoint {
    description: &'static str,
    router: Router,
}

impl OpenAiEndpoint {
    #[tracing::instrument]
    pub fn new<F: OpenAiRouterFactory>() -> Self {
        // Create the routes under /api/v1 to match OpenAi Platform endpoints
        let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
            .routes(routes!(health))
            .nest("/api/v1", F::routes())
            .split_for_parts();

        // API Documentation
        let router = router.merge(Redoc::with_url("/doc", api.clone()));

        Self {
            // api,
            router,
            description: F::description(),
        }
    }

    pub fn run<A: ToSocketAddrs>(self, interface: A) -> Result<(), ()> {
        let mut rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            println!("Spawning OpenAi Endpoint");
            let transport = TcpListener::bind(interface).await.expect("Failed to bind");
            let router = self.router;

            axum::serve(transport, router)
                .await
                .expect("Failed to serve");
        });

        Ok(())
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::openai::audio::transcription::TranscriptionEndpointFactory;
    use crate::openai::OpenAiEndpoint;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;

    #[pyclass(name = "TranscriptionEndpoint", subclass)]
    pub struct PyTranscriptionEndpoint {
        inner: Option<OpenAiEndpoint>,
        description: &'static str,
    }

    #[pymethods]
    impl PyTranscriptionEndpoint {
        #[new]
        pub fn new() -> Self {
            let inner = OpenAiEndpoint::new::<TranscriptionEndpointFactory>();
            let description = inner.description;

            Self {
                inner: Some(inner),
                description,
            }
        }

        pub fn description(&self) -> &'static str {
            self.description
        }

        pub fn run(&mut self, interface: String, port: u16) -> PyResult<()> {
            let inner = std::mem::take(&mut self.inner);
            if let Some(inner) = inner {
                inner.run((interface, port)).expect("Failed to run python");
            }
            Ok(())
        }
    }

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<PyTranscriptionEndpoint>()?
            .finish();

        Ok(module)
    }
}
