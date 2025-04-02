#[cfg(feature = "python")]
mod python {
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_openai as openai;
    use pyo3::prelude::*;

    pub const __VERSION__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule]
    pub fn _hfendpoints(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
        // Print Rust logs in Python's ones
        // pyo3_log::init();

        tracing_subscriber::fmt::init();

        let name = m.name()?.extract::<String>()?;

        // hfendpoints
        let pymodule_hfendpoints = ImportablePyModuleBuilder::from(m)
            .defaults()?
            // .add_submodule(&hfendpoints_core::python::bind(py, &name)?)?
            .add_submodule(&openai::python::bind(py, &format!("{name}.openai"))?)?
            .finish();

        pymodule_hfendpoints.add("__version__", __VERSION__)?;
        Ok(())
    }
}
