mod transcription;

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::audio::transcription;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::PyModule;
    use pyo3::{Bound, PyResult, Python};

    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_submodule(&transcription::python::bind(
                py,
                &format!("{name}.transcription"),
            )?)?
            .finish();

        Ok(module)
    }
}
