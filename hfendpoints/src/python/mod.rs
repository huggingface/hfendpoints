use crate::Endpoint;
use hfendpoints_transports_axum::python::register_hfendpoints_openai_pymodule;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Endpoint")]
pub struct PyEndpoint {}

impl Endpoint for PyEndpoint {}

#[pymodule(name = "hfendpoints")]
fn register_hfendpoints_pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEndpoint>()?;

    // hfendpoints.openai
    register_hfendpoints_openai_pymodule(m);
    Ok(())
}
