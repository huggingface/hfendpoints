use serde::Serialize;
use std::sync::atomic::AtomicU32;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(frozen))]
#[derive(Serialize)]
pub struct InFlightStats {
    in_flight: AtomicU32,
    in_queue: AtomicU32,
    max_in_flight: AtomicU32,
    max_in_queue: AtomicU32,
}
