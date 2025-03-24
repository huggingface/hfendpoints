/// High-level object containing all the information required to run the actual endpoint
pub trait Endpoint {
    /// Main entrypoint for running the actual endpoint
    fn run(&self) -> impl Future<Output = ()> + Send;
}

#[cfg(feature = "python")]
mod python {
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_transports_axum::openai;
    use pyo3::prelude::*;

    pub const __VERSION__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule]
    pub fn hfendpoints(py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
        let name = m.name()?.extract::<String>()?;

        // hfendpoints
        let pymodule_hfendpoints = ImportablePyModuleBuilder::from(m)
            .defaults()?
            .add_submodule(&openai::python::bind(py, &format!("{name}.openai"))?)?
            .finish();

        pymodule_hfendpoints.add("__version__", __VERSION__)?;
        Ok(())
    }
}
