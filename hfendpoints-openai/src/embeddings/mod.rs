#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::error::OpenAiError;
use crate::headers::RequestId;
use crate::{Context, OpenAiResult};
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum_extra::TypedHeader;
use hfendpoints_core::{EndpointContext, Error};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

pub const EMBEDDINGS_TAG: &str = "Embeddings";
pub const EMBEDDINGS_DESC: &str = "Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.";


#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Copy, Clone, Serialize, ToSchema)]
pub struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Embedding {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[cfg_attr(feature = "python", pymethods)]
impl Embedding {
    #[cfg_attr(feature = "python", new)]
    pub fn new(index: usize, embedding: Vec<f32>) -> Self {
        Self { object: "embedding", index, embedding }
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen, eq, eq_int))]
#[derive(Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct EmbeddingResponse {
    object: &'static str,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

#[cfg_attr(feature = "python", pymethods)]
impl EmbeddingResponse {
    #[cfg_attr(feature = "python", new)]
    pub fn new(data: Vec<Embedding>, model: String, usage: Usage) -> Self {
        Self { object: "list", data, model, usage }
    }
}

impl IntoResponse for EmbeddingResponse {
    fn into_response(self) -> Response {
        todo!()
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Deserialize, ToSchema)]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Deserialize, ToSchema)]
pub enum MaybeBatched<T>
where
    T: Clone + Sized,
{
    Single(T),
    Batch(Vec<T>),
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass)]
#[derive(Clone, Deserialize, ToSchema)]
pub struct EmbeddingRequest {
    input: MaybeBatched<EmbeddingInput>,
    model: Option<String>,
    dimension: Option<usize>,
    encoding_format: EncodingFormat,
    user: Option<String>,
}

#[utoipa::path(
    post,
    path = "/embeddings",
    tag = EMBEDDINGS_TAG,
    request_body(content = EmbeddingRequest, content_type = "application/json"),
    responses(
        (status = OK, description = "Creates an embedding vector representing the input text.", body = EmbeddingResponse),
    )
)]
#[instrument(skip(state, request))]
pub async fn embed(
    State(state): State<EndpointContext<(EmbeddingRequest, Context), EmbeddingResponse>>,
    request_id: TypedHeader<RequestId>,
    Json(request): Json<EmbeddingRequest>,
) -> OpenAiResult<EmbeddingResponse> {
    // Create request context
    let ctx = Context::new(request_id.0);

    // Ask for the inference thread to handle it and wait for answers
    let mut egress = state.schedule((request, ctx));
    if let Some(response) = egress.recv().await {
        Ok(response?)
    } else {
        Err(OpenAiError::NoResponse)
    }
}

/// Helper factory to build
/// [OpenAi Platform compatible Transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription)
#[derive(Clone)]
pub struct EmbeddingRouter(
    pub UnboundedSender<(
        (EmbeddingRequest, Context),
        UnboundedSender<Result<EmbeddingResponse, Error>>,
    )>,
);

impl From<EmbeddingRouter> for OpenApiRouter {
    fn from(value: EmbeddingRouter) -> Self {
        OpenApiRouter::new()
            .routes(routes!(embed))
            .with_state(EndpointContext::<(EmbeddingRequest, Context), EmbeddingResponse>::new(value.0))
    }
}

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::embeddings::{EmbeddingInput, EmbeddingRequest, EmbeddingResponse, EmbeddingRouter, EncodingFormat, MaybeBatched};
    use crate::python::{impl_pyendpoint, impl_pyhandler};
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::types::{PyList, PyString};

    #[pymethods]
    impl EmbeddingRequest {
        #[getter]
        pub fn encoding_format(&self) -> EncodingFormat {
            self.encoding_format
        }
        //
        // pub fn input(&self, py: Python<'_>) -> PyResult<PyObject> {
        //     let pyobj = match &self.input {
        //         MaybeBatched::Single(item) => match item {
        //             EmbeddingInput::Text(text) => text.into_py(py),
        //             EmbeddingInput::Tokens(tokens) => tokens.into_py(py),
        //         }
        //         MaybeBatched::Batch(items) => {
        //             items.into_py(py)
        //         }
        //     };
        //
        //     Ok(pyobj)
        // }
    }


    impl_pyhandler!(EmbeddingRequest, EmbeddingResponse);
    impl_pyendpoint!(
        "EmbeddingHandler",
        PyEmbeddingHandler,
        PyHandler,
        EmbeddingRouter
    );

    // Bind hfendpoints.openai.audio submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<EncodingFormat>()?
            .add_class::<EmbeddingRequest>()?
            .add_class::<EmbeddingResponse>()?
            .finish();

        Ok(module)
    }
}