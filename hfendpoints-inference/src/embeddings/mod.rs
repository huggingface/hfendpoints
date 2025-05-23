use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum_extra::TypedHeader;
use hfendpoints_core::{EndpointContext, EndpointResult, Error};
use hfendpoints_http::headers::RequestId;
use hfendpoints_http::{Context, HttpError, HttpResult, RequestWithContext, EMBEDDINGS_TAG};
use hfendpoints_tasks::embedding::{
    EmbeddingInput, EmbeddingParams, EmbeddingRequest, EmbeddingResponse,
};
use hfendpoints_tasks::MaybeBatched;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub struct HuggingFaceInferenceEmbeddingRequest {
    /// The text or list of texts to embed.
    inputs: MaybeBatched<EmbeddingInput>,

    #[serde(flatten)]
    parameters: EmbeddingParams,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
pub struct HuggingFaceInferenceEmbeddingResponse(MaybeBatched<Vec<f32>>);

type HuggingFaceInferenceEmbeddingRequestWithContext =
    RequestWithContext<HuggingFaceInferenceEmbeddingRequest>;

impl TryFrom<HuggingFaceInferenceEmbeddingRequest> for EmbeddingRequest {
    type Error = Error;

    #[inline]
    fn try_from(value: HuggingFaceInferenceEmbeddingRequest) -> Result<Self, Self::Error> {
        Ok(EmbeddingRequest::new(value.inputs, value.parameters))
    }
}

impl TryFrom<EmbeddingResponse> for HuggingFaceInferenceEmbeddingResponse {
    type Error = Error;

    #[inline]
    fn try_from(value: EmbeddingResponse) -> Result<Self, Self::Error> {
        Ok(Self(value.output))
    }
}

impl IntoResponse for HuggingFaceInferenceEmbeddingResponse {
    fn into_response(self) -> Response {
        Json::from(self).into_response()
    }
}

#[utoipa::path(
    post,
    path = "/embeddings",
    tag = EMBEDDINGS_TAG,
    request_body(content = HuggingFaceInferenceEmbeddingRequest, content_type = "application/json"),
    responses(
        (status = OK, description = "Creates an embedding vector representing the input text.", body = HuggingFaceInferenceEmbeddingResponse),
    )
)]
#[instrument(skip(state, request))]
async fn embed(
    State(state): State<
        EndpointContext<
            HuggingFaceInferenceEmbeddingRequestWithContext,
            HuggingFaceInferenceEmbeddingResponse,
        >,
    >,
    request_id: TypedHeader<RequestId>,
    Json(request): Json<HuggingFaceInferenceEmbeddingRequest>,
) -> HttpResult<HuggingFaceInferenceEmbeddingResponse> {
    let ctx = Context::new(request_id.0);

    let mut egress = state.schedule((request, ctx))?;
    if let Some(response) = egress.recv().await {
        Ok(response?)
    } else {
        Err(HttpError::NoResponse)
    }
}

#[derive(Clone)]
pub struct HuggingFaceInferenceEmbeddingRouter(
    pub  UnboundedSender<(
        HuggingFaceInferenceEmbeddingRequestWithContext,
        UnboundedSender<EndpointResult<HuggingFaceInferenceEmbeddingResponse>>,
    )>,
);

impl From<HuggingFaceInferenceEmbeddingRouter> for OpenApiRouter {
    fn from(value: HuggingFaceInferenceEmbeddingRouter) -> Self {
        OpenApiRouter::new()
            .routes(routes!(embed))
            .with_state(EndpointContext::new(value.0))
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::embeddings::{
        HuggingFaceInferenceEmbeddingRequest, HuggingFaceInferenceEmbeddingResponse,
        HuggingFaceInferenceEmbeddingRouter,
    };
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_http::{impl_http_pyendpoint, impl_http_pyhandler};
    use hfendpoints_tasks::embedding::python::{PyEmbeddingRequest, PyEmbeddingResponse};
    use pyo3::prelude::PyModule;
    use pyo3::{Bound, PyResult, Python};

    impl_http_pyhandler!(
        HuggingFaceInferenceEmbeddingRequest,
        HuggingFaceInferenceEmbeddingResponse,
        PyEmbeddingRequest,
        PyEmbeddingResponse
    );

    impl_http_pyendpoint!(
        "EmbeddingEndpoint",
        PyEmbeddingEndpoint,
        PyHandler,
        HuggingFaceInferenceEmbeddingRouter
    );

    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<PyEmbeddingEndpoint>()?
            .finish();

        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_conversion() {
        let request = HuggingFaceInferenceEmbeddingRequest {
            inputs: EmbeddingInput::Text("test text".to_owned()).into(),
            parameters: EmbeddingParams::default(),
        };

        let converted: EmbeddingRequest = request.try_into().unwrap();
        assert_eq!(
            converted.inputs,
            MaybeBatched::Single(EmbeddingInput::Text("test text".into()))
        );
    }

    #[test]
    fn test_embedding_response_conversion() {
        let embeddings = vec![0.1, 0.2, 0.3];
        let response = EmbeddingResponse {
            output: MaybeBatched::Single(embeddings.clone()),
            usage: None,
        };

        let converted: HuggingFaceInferenceEmbeddingResponse = response.try_into().unwrap();
        assert_eq!(converted.0, MaybeBatched::Single(embeddings));
    }

    #[test]
    fn test_batched_embedding_request_conversion() {
        let inputs = vec!["test1".to_owned(), "test2".to_owned()]
            .into_iter()
            .map(EmbeddingInput::Text)
            .collect::<Vec<_>>();
        let request = HuggingFaceInferenceEmbeddingRequest {
            inputs: inputs.clone().into(),
            parameters: EmbeddingParams::default(),
        };

        let converted: EmbeddingRequest = request.try_into().unwrap();
        assert_eq!(converted.inputs, MaybeBatched::Batch(inputs));
    }

    #[test]
    fn test_batched_embedding_response_conversion() {
        let embeddings = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let response = EmbeddingResponse {
            output: MaybeBatched::Batch(embeddings.clone()),
            usage: None,
        };

        let converted: HuggingFaceInferenceEmbeddingResponse = response.try_into().unwrap();
        assert_eq!(converted.0, MaybeBatched::Batch(embeddings));
    }
}
