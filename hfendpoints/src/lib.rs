#[cfg(feature = "python")]
mod python {
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_http as http;
    use hfendpoints_inference as hfinference;
    use hfendpoints_openai as openai;
    use hfendpoints_tasks as tasks;
    use pyo3::prelude::*;

    pub const __VERSION__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule]
    pub fn _hfendpoints(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
        tracing_subscriber::fmt::init();

        let name = m.name()?.extract::<String>()?;

        // hfendpoints
        let pymodule_hfendpoints = ImportablePyModuleBuilder::from(m)
            .defaults()?
            .add_submodule(&http::python::bind(py, &format!("{name}.http"))?)?
            .add_submodule(&hfinference::python::bind(
                py,
                &format!("{name}.hfinference"),
            )?)?
            .add_submodule(&tasks::python::bind(py, &format!("{name}.tasks"))?)?
            .add_submodule(&openai::python::bind(py, &format!("{name}.openai"))?)?
            .finish();

        pymodule_hfendpoints.add("__version__", __VERSION__)?;
        Ok(())
    }
}
