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
use hfendpoints_tasks::{MaybeBatched, Usage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
enum EmbeddingTag {
    Embedding,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Embedding {
    object: EmbeddingTag,
    index: usize,
    embedding: Vec<f32>,
}

impl Embedding {
    pub fn new(index: usize, embedding: Vec<f32>) -> Self {
        Self {
            object: EmbeddingTag::Embedding,
            index,
            embedding,
        }
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Copy, Clone, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
enum EmbeddingResponseTag {
    List,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
pub struct OpenAiEmbeddingResponse {
    object: EmbeddingResponseTag,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

impl OpenAiEmbeddingResponse {
    pub fn new(data: Vec<Embedding>, model: String, usage: Usage) -> Self {
        Self {
            object: EmbeddingResponseTag::List,
            data,
            model,
            usage,
        }
    }
}

impl IntoResponse for OpenAiEmbeddingResponse {
    #[inline]
    fn into_response(self) -> Response {
        Json::from(self).into_response()
    }
}

#[allow(dead_code)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Serialize))]
#[derive(Clone, Deserialize, ToSchema)]
pub struct OpenAiEmbeddingRequest {
    #[serde(default)]
    encoding_format: EncodingFormat,
    input: MaybeBatched<EmbeddingInput>,
    model: Option<String>,
    dimension: Option<usize>,
    user: Option<String>,
}

type OpenAiEmbeddingRequestWithContext = RequestWithContext<OpenAiEmbeddingRequest>;

#[utoipa::path(
    post,
    path = "/embeddings",
    tag = EMBEDDINGS_TAG,
    request_body(content = OpenAiEmbeddingRequest, content_type = "application/json"),
    responses(
        (status = OK, description = "Creates an embedding vector representing the input text.", body = OpenAiEmbeddingResponse),
    )
)]
#[instrument(skip(state, request))]
pub async fn embed(
    State(state): State<
        EndpointContext<OpenAiEmbeddingRequestWithContext, OpenAiEmbeddingResponse>,
    >,
    request_id: TypedHeader<RequestId>,
    Json(request): Json<OpenAiEmbeddingRequest>,
) -> HttpResult<OpenAiEmbeddingResponse> {
    // Create request context
    let ctx = Context::new(request_id.0);

    // Ask for the inference thread to handle it and wait for answers
    let mut egress = state.schedule((request, ctx))?;
    if let Some(response) = egress.recv().await {
        Ok(response?)
    } else {
        Err(HttpError::NoResponse)
    }
}

/// Helper factory to build
/// [HTTP Transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription)
#[derive(Clone)]
pub struct OpenAiEmbeddingRouter(
    pub  UnboundedSender<(
        OpenAiEmbeddingRequestWithContext,
        UnboundedSender<EndpointResult<OpenAiEmbeddingResponse>>,
    )>,
);

impl From<OpenAiEmbeddingRouter> for OpenApiRouter {
    fn from(value: OpenAiEmbeddingRouter) -> Self {
        OpenApiRouter::new()
            .routes(routes!(embed))
            .with_state(EndpointContext::new(value.0))
    }
}

impl TryFrom<OpenAiEmbeddingRequest> for EmbeddingRequest {
    type Error = Error;

    #[inline]
    fn try_from(value: OpenAiEmbeddingRequest) -> Result<Self, Self::Error> {
        Ok(Self::new(
            value.input,
            EmbeddingParams::new(Some(true), None, None, None),
        ))
    }
}

impl TryFrom<EmbeddingResponse> for OpenAiEmbeddingResponse {
    type Error = Error;

    fn try_from(value: EmbeddingResponse) -> Result<Self, Self::Error> {
        let usage = value.usage.unwrap_or_default();
        let embeddings = match value.output {
            MaybeBatched::Single(item) => vec![Embedding::new(0, item)],
            MaybeBatched::Batch(items) => items
                .into_iter()
                .enumerate()
                .map(|(index, item)| Embedding::new(index, item))
                .collect(),
        };

        Ok(Self::new(embeddings, String::new(), usage))
    }
}

#[cfg(feature = "python")]
pub mod python {
    use crate::embeddings::{
        OpenAiEmbeddingRequest, OpenAiEmbeddingResponse, OpenAiEmbeddingRouter,
    };
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_http::{impl_http_pyendpoint, impl_http_pyhandler};
    use hfendpoints_tasks::embedding::python::{PyEmbeddingRequest, PyEmbeddingResponse};

    impl_http_pyhandler!(
        OpenAiEmbeddingRequest,
        OpenAiEmbeddingResponse,
        PyEmbeddingRequest,
        PyEmbeddingResponse
    );

    impl_http_pyendpoint!(
        "EmbeddingEndpoint",
        PyEmbeddingEndpoint,
        PyHandler,
        OpenAiEmbeddingRouter
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
    use crate::embeddings::{
        embed, Embedding, EmbeddingResponseTag, EmbeddingTag,
        EncodingFormat, OpenAiEmbeddingRequestWithContext,
    };
    use axum::{
        body::Body,
        http::{self, Request, StatusCode},
        routing::post,
        Router,
    };
    use hfendpoints_core::{EndpointContext, EndpointResult, Error};
    use hfendpoints_tasks::embedding::{EmbeddingInput, EmbeddingResponse};
    use hfendpoints_tasks::{MaybeBatched, Usage};
    use http_body_util::BodyExt;
    use hyper::body::Buf;
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use tower::util::ServiceExt;
    use tower_http::timeout::TimeoutLayer;

    // Test helper to create a test app
    fn create_test_app(
        sender: UnboundedSender<(
            OpenAiEmbeddingRequestWithContext,
            UnboundedSender<EndpointResult<OpenAiEmbeddingResponse>>,
        )>,
    ) -> Router {
        let state = EndpointContext::new(sender);
        Router::new()
            .route("/embeddings", post(embed))
            .with_state(state)
            .layer(TimeoutLayer::new(Duration::from_secs(5)))
    }

    #[tokio::test]
    async fn test_successful_embedding() {
        // Create channels for test communication
        let (tx, mut rx) = unbounded_channel();
        let app = create_test_app(tx);

        // Create a test request
        let request = OpenAiEmbeddingRequest {
            input: MaybeBatched::Single(EmbeddingInput::Text(String::from("test text"))),
            model: Some("test-model".into()),
            dimension: None,
            encoding_format: EncodingFormat::Float,
            user: None,
        };

        // Create a test response
        let response = OpenAiEmbeddingResponse {
            object: EmbeddingResponseTag::List,
            data: vec![Embedding {
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
                object: EmbeddingTag::Embedding,
            }],
            model: "test-model".into(),
            usage: Usage {
                prompt_tokens: 1,
                total_tokens: 1,
            },
        };

        // Build the HTTP request
        let request = Request::builder()
            .uri("/embeddings")
            .method(http::Method::POST)
            .header("content-type", "application/json")
            .header("x-request-id", "test-request-id")
            .body(Body::from(serde_json::to_string(&request).unwrap()))
            .unwrap();

        // Spawn a task to handle the response
        tokio::spawn(async move {
            if let Some((_, sender)) = rx.recv().await {
                sender.send(Ok(response)).unwrap();
            }
        });

        // Send the request and check the response
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Verify response body
        let body = response.collect().await.unwrap().aggregate();
        let response_json: OpenAiEmbeddingResponse =
            serde_json::from_reader(body.reader()).unwrap();
        assert_eq!(response_json.model, "test-model");
        assert_eq!(response_json.data.len(), 1);
    }

    #[tokio::test]
    async fn test_missing_request_id() {
        let (tx, _) = unbounded_channel();
        let app = create_test_app(tx);

        let request = Request::builder()
            .uri("/embeddings")
            .method(http::Method::POST)
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&json!({
                    "input": "test",
                    "model": "test-model"
                }))
                .unwrap(),
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    // TODO: reenable this one as currently it returns 422 instead of 408
    // #[tokio::test]
    // async fn test_no_response_from_handler() {
    //     let (tx, rx) = unbounded_channel();
    //     let app = create_test_app(tx);
    //
    //     let request = Request::builder()
    //         .uri("/embeddings")
    //         .method(http::Method::POST)
    //         .header("content-type", "application/json")
    //         .header("x-request-id", "test-request-id")
    //         .body(Body::from(
    //             serde_json::to_string(&json!({
    //                 "input": "test",
    //                 "model": "test-model"
    //             }))
    //             .unwrap(),
    //         ))
    //         .unwrap();
    //
    //     // Spawn a task that sends an error response
    //     tokio::spawn(async move {
    //         sleep(Duration::from_secs(6)).await;
    //     });
    //
    //     let response = app.oneshot(request).await.unwrap();
    //     assert!(rx.is_empty());
    //     assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    // }

    #[tokio::test]
    async fn test_invalid_request_body() {
        let (tx, _rx) = unbounded_channel();
        let app = create_test_app(tx);

        let request = Request::builder()
            .uri("/embeddings")
            .method(http::Method::POST)
            .header("content-type", "application/json")
            .header("x-request-id", "test-request-id")
            .body(Body::from("invalid json"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_handler_error() {
        let (tx, mut rx) = unbounded_channel();
        let app = create_test_app(tx);

        let request = OpenAiEmbeddingRequest {
            input: MaybeBatched::Single(EmbeddingInput::Text("test text".into())),
            model: Some("test-model".into()),
            dimension: None,
            encoding_format: EncodingFormat::Float,
            user: None,
        };

        let request = Request::builder()
            .uri("/embeddings")
            .method(http::Method::POST)
            .header("content-type", "application/json")
            .header("x-request-id", "test-request-id")
            .body(Body::from(serde_json::to_string(&request).unwrap()))
            .unwrap();

        // Spawn a task that sends an error response
        tokio::spawn(async move {
            if let Some((_, sender)) = rx.recv().await {
                sender.send(Err(Error::TestOnly("Test error"))).unwrap();
            }
        });

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_embedding_response_to_openai_conversion_single() {
        // Test single embedding conversion
        let single_response = EmbeddingResponse {
            output: MaybeBatched::Single(vec![0.1, 0.2, 0.3]),
            usage: Some(Usage::new(1, 2)),
        };

        let converted = OpenAiEmbeddingResponse::try_from(single_response).unwrap();
        assert_eq!(converted.data.len(), 1);
        assert_eq!(converted.data[0].embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.usage.prompt_tokens, 1);
        assert_eq!(converted.usage.total_tokens, 2);

        // Test usage conversion
        let response_without_usage = EmbeddingResponse {
            output: MaybeBatched::Single(vec![0.1]),
            usage: None,
        };

        let converted = OpenAiEmbeddingResponse::try_from(response_without_usage).unwrap();
        assert_eq!(converted.data.len(), 1);
        assert_eq!(converted.data[0].embedding, vec![0.1]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.usage.prompt_tokens, 0);
        assert_eq!(converted.usage.total_tokens, 0);
    }

    #[test]
    fn test_embedding_response_to_openai_conversion_single_no_usage() {
        // Test single embedding conversion
        let response_without_usage = EmbeddingResponse {
            output: MaybeBatched::Single(vec![0.1]),
            usage: None,
        };

        let converted = OpenAiEmbeddingResponse::try_from(response_without_usage).unwrap();
        assert_eq!(converted.data.len(), 1);
        assert_eq!(converted.data[0].embedding, vec![0.1]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.usage.prompt_tokens, 0);
        assert_eq!(converted.usage.total_tokens, 0);
    }

    #[test]
    fn test_embedding_response_to_openai_conversion_batched() {
        // Test batched embeddings conversion
        let batched_response = EmbeddingResponse {
            output: MaybeBatched::Batch(vec![vec![0.1, 0.2], vec![0.3, 0.4]]),
            usage: Some(Usage::new(2, 3)),
        };

        let converted = OpenAiEmbeddingResponse::try_from(batched_response).unwrap();
        assert_eq!(converted.data.len(), 2);
        assert_eq!(converted.data[0].embedding, vec![0.1, 0.2]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.data[1].embedding, vec![0.3, 0.4]);
        assert_eq!(converted.data[1].index, 1);
        assert_eq!(converted.usage.prompt_tokens, 2);
        assert_eq!(converted.usage.total_tokens, 3);

        // Test usage conversion
        let response_without_usage = EmbeddingResponse {
            output: MaybeBatched::Batch(vec![vec![0.1, 0.2], vec![0.3, 0.4]]),
            usage: None,
        };

        let converted = OpenAiEmbeddingResponse::try_from(response_without_usage).unwrap();
        assert_eq!(converted.data.len(), 2);
        assert_eq!(converted.data[0].embedding, vec![0.1, 0.2]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.data[1].embedding, vec![0.3, 0.4]);
        assert_eq!(converted.data[1].index, 1);
        assert_eq!(converted.usage.prompt_tokens, 0);
        assert_eq!(converted.usage.total_tokens, 0);
    }

    #[test]
    fn test_embedding_response_to_openai_conversion_batched_no_usage() {
        // Test batched embeddings conversion
        let response_without_usage = EmbeddingResponse {
            output: MaybeBatched::Batch(vec![vec![0.1, 0.2], vec![0.3, 0.4]]),
            usage: None,
        };

        let converted = OpenAiEmbeddingResponse::try_from(response_without_usage).unwrap();
        assert_eq!(converted.data.len(), 2);
        assert_eq!(converted.data[0].embedding, vec![0.1, 0.2]);
        assert_eq!(converted.data[0].index, 0);
        assert_eq!(converted.data[1].embedding, vec![0.3, 0.4]);
        assert_eq!(converted.data[1].index, 1);
        assert_eq!(converted.usage.prompt_tokens, 0);
        assert_eq!(converted.usage.total_tokens, 0);
    }
}
