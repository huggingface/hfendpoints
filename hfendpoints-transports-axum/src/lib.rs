pub mod openai;

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    pub fn register_hfendpoints_openai_pymodule(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let openai_module = PyModule::new(m.py(), "openai")?;
        m.add_submodule(&openai_module)?;
        Ok(())
    }
}
