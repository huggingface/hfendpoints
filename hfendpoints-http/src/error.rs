use axum::extract::multipart::MultipartError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use hfendpoints_core::Error as EndpointError;
use std::num::ParseFloatError;
use thiserror::Error;
use tokio::io::Error as TokioIoError;

/// Define all the possible errors for OpenAI Compatible Endpoint
#[derive(Debug, Error)]
pub enum HttpError {
    #[error("Endpoint error: {0}")]
    Endpoint(#[from] EndpointError),

    #[error("I/O Error occurred: {0}")]
    Io(#[from] TokioIoError),

    #[error("Malformed multipart/form-data payload: {0}")]
    Multipart(#[from] MultipartError),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("No response was returned by the inference engine")]
    NoResponse,
}

impl From<ParseFloatError> for HttpError {
    #[inline]
    fn from(value: ParseFloatError) -> Self {
        Self::Validation(value.to_string())
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            Self::Endpoint(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            Self::Io(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            Self::Validation(e) => (StatusCode::BAD_REQUEST, e),
            Self::Multipart(e) => (StatusCode::BAD_REQUEST, e.to_string()),
            Self::NoResponse => (
                StatusCode::INTERNAL_SERVER_ERROR,
                String::from("No response returned by the inference engine"),
            ),
        };

        (status, body).into_response()
    }
}
