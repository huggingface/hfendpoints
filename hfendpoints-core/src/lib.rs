mod context;
mod endpoint;
mod handler;
mod metrics;

pub use context::EndpointContext;
pub use endpoint::Endpoint;
pub use handler::{spawn_handler, Handler};
pub use metrics::InFlightStats;

#[cfg(feature = "python")]
use pyo3::PyErr;
use thiserror::Error;

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Error)]
pub enum Error {
    #[cfg(feature = "python")]
    #[error("Caught error while execution Python code: {0}")]
    PythonError(#[from] PyErr),
}
