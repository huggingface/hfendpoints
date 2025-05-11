mod context;
mod endpoint;
mod handler;
mod metrics;

pub use crate::handler::HandlerError;
pub use context::EndpointContext;
pub use endpoint::Endpoint;
pub use handler::{wait_for_requests, Handler};
pub use metrics::InFlightStats;
use thiserror::Error;

#[cfg(feature = "python")]
use pyo3::PyErr;

#[derive(Debug, Error)]
pub enum Error {
    #[cfg(feature = "python")]
    #[error("Caught error while executing Python code: {0}")]
    Python(#[from] PyErr),

    #[error("{0}")]
    Handler(#[from] HandlerError),

    #[cfg(debug_assertions)]
    #[error("{0}")]
    TestOnly(&'static str),
}

/// Result with predefined hfendpoints-core::Error as the Error type
pub type EndpointResult<T> = Result<T, Error>;
