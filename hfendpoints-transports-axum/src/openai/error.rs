use crate::openai::error::OpenAiError::ValidationError;
use axum::extract::multipart::MultipartError;
use axum::response::{IntoResponse, Response};
use std::num::ParseFloatError;
use thiserror::Error;
use tokio::io::Error as TokioIoError;

/// Define all the possible errors for OpenAI Compatible Endpoint
#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("I/O Error occured: {0}")]
    IoError(#[from] TokioIoError),

    #[error("Malformed multipart/form-data payload: {0}")]
    MultipartError(#[from] MultipartError),

    #[error("Validation failed: {0}")]
    ValidationError(String),
}

impl From<ParseFloatError> for OpenAiError {
    #[inline]
    fn from(value: ParseFloatError) -> Self {
        ValidationError(value.to_string())
    }
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let builder = match self {
            OpenAiError::IoError(msg) => Response::builder().status(500).body(msg.to_string()),
            OpenAiError::MultipartError(msg) => {
                Response::builder().status(400).body(msg.to_string())
            }
            ValidationError(msg) => Response::builder().status(403).body(msg),
        };

        builder.unwrap().into_response()
    }
}
