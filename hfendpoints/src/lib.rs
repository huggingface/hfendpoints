#[cfg(feature = "python")]
mod python {
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_http as http;
    use hfendpoints_io as io;
    use hfendpoints_openai as openai;
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
            .add_submodule(&io::python::bind(py, &format!("{name}.io"))?)?
            .add_submodule(&openai::python::bind(py, &format!("{name}.openai"))?)?
            .finish();

        pymodule_hfendpoints.add("__version__", __VERSION__)?;
        Ok(())
    }
}
