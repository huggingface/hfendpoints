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
#[cfg_attr(test, derive(Deserialize))]
#[derive(Copy, Clone, Serialize, ToSchema)]
pub struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
enum EmbeddingTag {
    Embedding,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen))]
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
#[cfg_attr(feature = "python", pyclass(frozen, eq, eq_int))]
#[derive(Clone, Copy, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Copy, Clone, Serialize, ToSchema)]
#[serde(rename_all = "lowercase")]
enum EmbeddingResponseTag {
    List,
}

#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Deserialize))]
#[derive(Clone, Serialize, ToSchema)]
pub struct EmbeddingResponse {
    object: EmbeddingResponseTag,
    data: Vec<Embedding>,
    model: String,
    usage: Usage,
}

impl EmbeddingResponse {
    pub fn new(data: Vec<Embedding>, model: String, usage: Usage) -> Self {
        Self {
            object: EmbeddingResponseTag::List,
            data,
            model,
            usage,
        }
    }
}

impl IntoResponse for EmbeddingResponse {
    #[inline]
    fn into_response(self) -> Response {
        Json::from(self).into_response()
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Serialize))]
#[cfg_attr(feature = "python", derive(IntoPyObjectRef))]
#[derive(Clone, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Serialize))]
#[derive(Clone, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MaybeBatched<T>
where
    T: Clone + Sized,
{
    Single(T),
    Batch(Vec<T>),
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Serialize))]
#[cfg_attr(feature = "python", pyclass(frozen, sequence))]
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
    pub  UnboundedSender<(
        (EmbeddingRequest, Context),
        UnboundedSender<Result<EmbeddingResponse, Error>>,
    )>,
);

impl From<EmbeddingRouter> for OpenApiRouter {
    fn from(value: EmbeddingRouter) -> Self {
        OpenApiRouter::new()
            .routes(routes!(embed))
            .with_state(EndpointContext::<
                (EmbeddingRequest, Context),
                EmbeddingResponse,
            >::new(value.0))
    }
}

#[cfg(feature = "python")]
pub(crate) mod python {
    use pyo3::exceptions::PyIndexError;
    use pyo3::types::PyList;
    use crate::embeddings::{
        Embedding, EmbeddingRequest, EmbeddingResponse, EmbeddingResponseTag,
        EmbeddingRouter, EmbeddingTag, EncodingFormat, MaybeBatched, Usage,
    };
    use crate::python::{impl_pyendpoint, impl_pyhandler};
    use hfendpoints_binding_python::ImportablePyModuleBuilder;

    #[pymethods]
    impl Usage {
        #[new]
        fn py_new(prompt_tokens: usize, total_tokens: usize) -> Self {
            Self {
                prompt_tokens,
                total_tokens,
            }
        }

        #[getter]
        fn prompt_tokens(&self) -> usize {
            self.prompt_tokens
        }

        #[getter]
        fn total_tokens(&self) -> usize {
            self.total_tokens
        }
    }

    #[pymethods]
    impl Embedding {
        #[new]
        fn py_new(index: u32, embedding: Bound<PyList>) -> PyResult<Self> {
            Ok(Self {
                object: EmbeddingTag::Embedding,
                index: index as usize,
                embedding: embedding.extract::<Vec<f32>>()?,
            })
        }
    }

    #[pymethods]
    impl EmbeddingRequest {

        pub fn __len__(&self) -> usize {
            match &self.input {
                MaybeBatched::Single(_) => 1,
                MaybeBatched::Batch(items) => items.len()
            }
        }

        pub fn __get_item__<'py>(&self, py: Python<'py>, index: usize)-> PyResult<Bound<'py, PyAny>> {
            match &self.input {
                MaybeBatched::Single(item) => {
                    if index == 0 {
                        item.into_pyobject(py)
                    } else {
                        Err(PyErr::new::<PyIndexError, _>("index out of range"))
                    }
                }
                MaybeBatched::Batch(items) => {
                    let item = items.get(index).ok_or(PyErr::new::<PyIndexError, _>("index out of range"))?;
                    item.into_pyobject(py)
                }
            }
        }

        #[getter]
        pub fn encoding_format(&self) -> EncodingFormat {
            self.encoding_format
        }

        #[getter]
        pub fn is_batched(&self) -> bool {
            match self.input {
                MaybeBatched::Single(_) => false,
                MaybeBatched::Batch(_) => true,
            }
        }

        #[getter]
        pub fn input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            match &self.input {
                MaybeBatched::Single(item) => item.into_pyobject(py),
                MaybeBatched::Batch(items) => items.into_pyobject(py),
            }
        }
    }

    #[pymethods]
    impl EmbeddingResponse {
        #[new]
        fn py_new(model: String, embeddings: Vec<Embedding>, usage: Usage) -> Self {
            Self {
                object: EmbeddingResponseTag::List,
                data: embeddings,
                model,
                usage
            }
        }
    }

    impl_pyhandler!(EmbeddingRequest, EmbeddingResponse);
    impl_pyendpoint!(
        "EmbeddingEndpoint",
        PyEmbeddingEndpoint,
        PyHandler,
        EmbeddingRouter
    );

    // Bind hfendpoints.openai.embeddings submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<Embedding>()?
            .add_class::<EncodingFormat>()?
            .add_class::<EmbeddingRequest>()?
            .add_class::<EmbeddingResponse>()?
            .add_class::<PyEmbeddingEndpoint>()?
            .add_class::<Usage>()?
            .finish();

        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{self, Request, StatusCode},
        routing::post,
        Router,
    };
    use http_body_util::BodyExt;
    use hyper::body::Buf;
    use serde_json::json;
    use std::time::Duration;
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use tower::ServiceExt;
    use tower_http::timeout::TimeoutLayer;

    // Test helper to create a test app
    fn create_test_app(
        sender: UnboundedSender<(
            (EmbeddingRequest, Context),
            UnboundedSender<Result<EmbeddingResponse, Error>>,
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

        // Create test request
        let request = EmbeddingRequest {
            input: MaybeBatched::Single(EmbeddingInput::Text(String::from("test text"))),
            model: Some("test-model".into()),
            dimension: None,
            encoding_format: EncodingFormat::Float,
            user: None,
        };

        // Create test response
        let response = EmbeddingResponse {
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
        let response_json: EmbeddingResponse = serde_json::from_reader(body.reader()).unwrap();
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

        let request = EmbeddingRequest {
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
                sender.send(Err(Error::TestError("Test error"))).unwrap();
            }
        });

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
