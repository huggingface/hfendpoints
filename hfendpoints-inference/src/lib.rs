mod embeddings;

#[cfg(feature = "python")]
pub mod python {
    use crate::embeddings;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_submodule(&embeddings::python::bind(py, &format!("{name}.embedding"))?)?
            .finish();

        Ok(module)
    }
}
