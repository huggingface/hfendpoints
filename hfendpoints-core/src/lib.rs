mod context;
mod endpoint;
mod handler;
mod metrics;

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
    PythonError(#[from] PyErr),

    #[error("{0}")]
    TestError(&'static str),
}


/// Result with predefined hfendpoints-core::Error as the Error type
pub type EndpointResult<T> = Result<T, Error>;