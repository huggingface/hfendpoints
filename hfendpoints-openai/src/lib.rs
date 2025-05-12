pub(crate) mod audio;
pub(crate) mod embeddings;

pub use hfendpoints_http::Context;

#[cfg(feature = "python")]
pub mod python {
    use hfendpoints_binding_python::tokio::create_multithreaded_runtime;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use hfendpoints_http::python::serve;
    use pyo3::prelude::*;
    use pyo3::prepare_freethreaded_python;
    use pyo3_async_runtimes::tokio::init;
    use tracing::instrument;

    #[pyfunction]
    #[instrument(skip(endpoint))]
    #[pyo3(name = "run")]
    fn run(endpoint: PyObject, interface: String, port: u16) -> PyResult<()> {
        prepare_freethreaded_python();

        // Initialize the tokio runtime and bind this runtime to the tokio <> asyncio compatible layer
        init(create_multithreaded_runtime());

        Python::with_gil(|py| {
            py.allow_threads(|| {
                pyo3_async_runtimes::tokio::get_runtime().block_on(async {
                    Python::with_gil(|inner| {
                        pyo3_async_runtimes::tokio::run(inner, serve(endpoint, interface, port))
                    })?;
                    Ok::<_, PyErr>(())
                })
            })
        })?;

        Ok::<_, PyErr>(())
    }

    /// Bind hfendpoints.openai submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .finish();

        module.add_function(wrap_pyfunction!(run, &module)?)?;
        Ok(module)
    }
}
