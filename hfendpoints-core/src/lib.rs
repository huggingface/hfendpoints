mod context;
mod endpoint;
mod handler;
mod metrics;

pub use context::EndpointContext;
pub use endpoint::Endpoint;
pub use handler::{wait_for_requests, Handler};
pub use metrics::InFlightStats;

#[cfg(feature = "python")]
use pyo3::PyErr;
use thiserror::Error;

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Error)]
pub enum Error {
    #[cfg(feature = "python")]
    #[error("Caught error while executing Python code: {0}")]
    PythonError(#[from] PyErr),
}
