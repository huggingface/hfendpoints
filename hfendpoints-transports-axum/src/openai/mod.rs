use axum::Router;
use thiserror::Error;
use tokio::io::Error as TokioIoError;
use tokio::net::ToSocketAddrs;
use utoipa::openapi::OpenApi;
use utoipa_axum::router::OpenApiRouter;

mod audio;

/// Define all the possible errors for OpenAI Compatible Endpoint
#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("I/O Error occured: {0}")]
    IoError(#[from] TokioIoError),
}

pub(crate) trait OpenAiRouterFactory {
    fn description() -> &'static str;
    fn routes() -> OpenApiRouter;
}

pub struct OpenAiEndpoint {
    description: &'static str,
    api: OpenApi,
    router: Router,
}

impl OpenAiEndpoint {
    pub fn new<F: OpenAiRouterFactory>() -> Self {
        // Create the routes under /api/v1 to match OpenAi Platform endpoints
        let (router, api) = OpenApiRouter::new()
            .nest("/api/v1", F::routes())
            .split_for_parts();

        Self {
            api,
            router,
            description: F::description(),
        }
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
        inner: OpenAiEndpoint,
    }

    #[pymethods]
    impl PyTranscriptionEndpoint {
        #[new]
        pub fn new() -> Self {
            Self {
                inner: OpenAiEndpoint::new::<TranscriptionEndpointFactory>(),
            }
        }

        pub fn description(&self) -> &'static str {
            self.inner.description
        }
    }

    /// Inject hfendpoints.openai submodule
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<PyTranscriptionEndpoint>()?
            .finish();

        Ok(module)
    }
}
