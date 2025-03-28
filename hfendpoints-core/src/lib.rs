mod endpoint;
mod handler;

use std::fmt::Debug;

#[cfg(feature = "python")]
pub mod python {
    // use hfendpoints_binding_python::ImportablePyModuleBuilder;
    // use pyo3::prelude::*;
    //
    // #[pyclass(name = "Handler", subclass)]
    // pub struct PyHandler {
    //     root: String,
    // }
    //
    // #[pymethods]
    // impl PyHandler {
    //     #[new]
    //     #[pyo3(signature = (model_id_or_path))]
    //     fn new(model_id_or_path: String) -> PyResult<Self> {
    //         Ok(Self {
    //             root: model_id_or_path,
    //         })
    //     }
    //
    //     pub async fn __call__(&mut self, _request: PyObject) -> PyResult<PyObject> {
    //         unimplemented!("PyHandler.__call__ is abstract and must be implemented.")
    //     }
    // }
    //
    // impl Handler for PyHandler {
    //     type Request = PyObject;
    //     type Response = PyResult<PyObject>;
    //
    //     async fn on_request(&mut self, request: Self::Request) -> Self::Response {
    //         self.__call__(request).await
    //     }
    // }
    //
    // pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    //     let module = ImportablePyModuleBuilder::new(py, name)?
    //         .defaults()?
    //         .add_class::<PyHandler>()?
    //         .finish();
    //
    //     Ok(module)
    // }
}
