use crate::headers::RequestId;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Holds the context in which a request is being executed
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen))]
#[derive(Clone)]
pub struct Context {
    /// Correlation ID for the current request
    request_id: RequestId,
}

impl Context {
    pub fn new(request_id: RequestId) -> Self {
        Self { request_id }
    }
}

#[cfg(feature = "python")]
mod python {
    use crate::context::Context;
    use pyo3::pymethods;

    #[pymethods]
    impl Context {
        #[getter]
        fn request_id(&self) -> &str {
            &self.request_id
        }
    }
}
