use crate::{EndpointRequest, EndpointResponse, MaybeBatched, Usage};

/// Represents a request to compute embeddings
pub type EmbeddingRequest = EndpointRequest<MaybeBatched<String>, ()>;

/// Represent a response to
pub type EmbeddingResponse = EndpointResponse<MaybeBatched<Vec<f32>>, Usage>;

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::embeddings::{EmbeddingRequest, EmbeddingResponse};
    use crate::{EndpointResponse, MaybeBatched, Usage};
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use pyo3::IntoPyObjectExt;
    use std::ops::Index;

    #[pyclass(name = "EmbeddingRequest", frozen)]
    pub struct PyEmbeddingRequest(EmbeddingRequest);

    #[pymethods]
    impl PyEmbeddingRequest {
        ///
        #[getter]
        fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            match &self.0.inputs {
                MaybeBatched::Single(item) => item.into_bound_py_any(py),
                MaybeBatched::Batched(items) => items.into_bound_py_any(py),
            }
        }

        #[getter]
        fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            py.None().into_bound_py_any(py)
        }
    }

    #[derive(FromPyObject)]
    enum SupportedEmbeddingsArray<'py> {
        Single(Bound<'py, PyArray1<f32>>),
        Batched(Bound<'py, PyArray2<f32>>),
    }

    #[pyclass(name = "EmbeddingResponse", frozen)]
    pub struct PyEmbeddingResponse(EmbeddingResponse);

    #[pymethods]
    impl PyEmbeddingResponse {
        /// Create an EmbeddingResponse by copying a Python's heap-allocated list into Rust's heap
        ///
        /// # Arguments
        ///
        /// * `embeddings`: Python's heap-allocated list of float
        /// * `num_tokens`: Number of tokens
        ///
        /// Returns: Result<PyEmbeddingResponse, PyErr>
        #[new]
        fn new<'py>(embeddings: Bound<'py, PyList>, num_tokens: usize) -> PyResult<Self> {
            Ok(Self {
                0: EndpointResponse {
                    output: embeddings.extract()?,
                    usage: Some(Usage::same(num_tokens)),
                },
            })
        }

        /// Create an EmbeddingResponse by pointing the internal Rust heap-allocated Vec to
        /// the same memory location as the provided ndarray.
        ///
        /// It is important to note, nothing is done to enforce the lifetime of the numpy array
        /// through this function. It is the caller's responsibility to ensure the array lives long enough.
        ///
        /// # Arguments
        ///
        /// * `embeddings`: Python's NumPy 1d-ndarray of float32
        /// * `num_tokens`: Number of tokens
        ///
        /// Returns: Result<PyEmbeddingResponse, PyErr>
        #[staticmethod]
        unsafe fn from_numpy<'py>(
            embeddings: SupportedEmbeddingsArray,
            num_tokens: usize,
        ) -> PyResult<Self> {
            //TODO(mfuntowicz) This does a copy for now
            let output = match embeddings {
                SupportedEmbeddingsArray::Single(item) => MaybeBatched::Single(item.to_vec()?),
                SupportedEmbeddingsArray::Batched(items) => {
                    let hidden = items.dims()[1];
                    let buffer = items.to_vec()?;
                    MaybeBatched::Batched(
                        buffer
                            .chunks_exact(hidden)
                            .map(|slice| slice.to_vec())
                            .collect(),
                    )
                }
            };

            Ok(Self {
                0: EndpointResponse {
                    output,
                    usage: Some(Usage::same(num_tokens)),
                },
            })
        }

        fn __repr__(&self) -> String {
            match &self.0.output {
                MaybeBatched::Single(single) => {
                    format!("EmbeddingResponse(<{}xf32>)", single.len())
                }
                MaybeBatched::Batched(batched) => format!(
                    "EmbeddingResponse(<{}x{}xf32>)",
                    batched.len(),
                    batched.get(0).map_or(0, |item| item.len())
                ),
            }
        }
    }

    /// Bind this module to the python's wheel  
    ///
    /// # Arguments
    ///
    /// * `py`: Python acquired GIL reference
    /// * `name`: name of the python module to register this under
    ///
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<PyEmbeddingRequest>()?
            .add_class::<PyEmbeddingResponse>()?
            .finish();

        Ok(module)
    }
}
