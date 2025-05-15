use crate::{EndpointRequest, EndpointResponse, MaybeBatched, Usage};
use hfendpoints_core::Handler;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[cfg(feature = "python")]
use pyo3::prelude::{IntoPyObject, IntoPyObjectRef};

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(IntoPyObjectRef))]
#[derive(Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
#[derive(PartialEq)]
pub enum EmbeddingInput {
    Text(String),
    Tokens(Vec<u32>),
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(IntoPyObject))]
#[derive(Copy, Clone, Default, ToSchema)]
pub struct EmbeddingParams {
    normalize: bool,
}

impl EmbeddingParams {
    pub fn new(normalize: bool) -> Self {
        Self { normalize }
    }
}

/// Represents a request to compute embeddings
pub type EmbeddingRequest = EndpointRequest<MaybeBatched<EmbeddingInput>, EmbeddingParams>;

/// Represent a response to
pub type EmbeddingResponse = EndpointResponse<MaybeBatched<Vec<f32>>, Usage>;

/// Helper trait to implement `Handler` specification for Transcription endpoints
pub trait EmbeddingHandler:
    Handler<Request = Self::TypedRequest, Response = Self::TypedResponse>
{
    type TypedRequest: Into<EmbeddingRequest>;
    type TypedResponse: From<EmbeddingResponse>;
}

#[cfg(feature = "python")]
pub mod python {
    use crate::embedding::{EmbeddingRequest, EmbeddingResponse};
    use crate::{EndpointResponse, MaybeBatched, Usage};
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use pyo3::IntoPyObjectExt;

    #[pyclass(name = "EmbeddingRequest", frozen)]
    pub struct PyEmbeddingRequest(pub EmbeddingRequest);

    #[pymethods]
    impl PyEmbeddingRequest {
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

    #[derive(Clone)]
    #[pyclass(name = "EmbeddingResponse", frozen)]
    pub struct PyEmbeddingResponse(pub EmbeddingResponse);

    #[pymethods]
    impl PyEmbeddingResponse {
        /// Create an EmbeddingResponse by copying a Python's heap-allocated list into Rust's heapxc
        ///
        /// # Arguments
        ///
        /// * `embeddings`: Python's heap-allocated list of float
        /// * `num_tokens`: Number of tokens
        ///
        /// Returns: Result<PyEmbeddingResponse, PyErr>
        #[new]
        fn new(embeddings: Bound<'_, PyList>, num_tokens: usize) -> PyResult<Self> {
            Ok(Self(EndpointResponse {
                output: embeddings.extract()?,
                usage: Some(Usage::same(num_tokens)),
            }))
        }

        /// Create an EmbeddingResponse by pointing the internal Rust heap-allocated Vec to
        /// the same memory location as the provided ndarray.
        ///
        /// It is important to note, nothing is done to enforce the lifetime of the numpy array
        /// through this function. It is the caller's responsibility to ensure the array lives long enough.
        ///
        /// # Arguments
        ///
        /// * `embeddings`: Python's NumPy 1d or 2d ndarray of float32
        /// * `num_tokens`: Number of tokens
        ///
        /// Returns: Result<PyEmbeddingResponse, PyErr>
        #[staticmethod]
        unsafe fn from_numpy(
            embeddings: SupportedEmbeddingsArray,
            num_tokens: usize,
        ) -> PyResult<Self> {
            let output = match embeddings {
                SupportedEmbeddingsArray::Single(item) => unsafe {
                    MaybeBatched::Single(Vec::from_raw_parts(item.data(), item.len(), item.len()))
                },

                //TODO(mfuntowicz) This does a copy for now
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

            Ok(Self(EndpointResponse {
                output,
                usage: Some(Usage::same(num_tokens)),
            }))
        }

        fn __repr__(&self) -> String {
            match &self.0.output {
                MaybeBatched::Single(single) => {
                    format!("EmbeddingResponse(<{}xf32>)", single.len())
                }
                MaybeBatched::Batched(batched) => format!(
                    "EmbeddingResponse(<{}x{}xf32>)",
                    batched.len(),
                    batched.first().map_or(0, |item| item.len())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MaybeBatched;

    #[test]
    fn test_single_embedding_request() {
        let input = "test text".to_string();
        let request = EmbeddingRequest {
            inputs: MaybeBatched::Single(EmbeddingInput::Text(input.clone())),
            parameters: EmbeddingParams::default(),
        };

        match request.inputs {
            MaybeBatched::Single(text) => match text {
                EmbeddingInput::Text(text) => assert_eq!(text, input),
                _ => panic!("Expected Text variant"),
            },
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_batched_embedding_request() {
        let inputs = ["text1".to_string(), "text2".to_string()];
        let request = EmbeddingRequest {
            inputs: MaybeBatched::Batched(
                inputs
                    .iter()
                    .map(|item| EmbeddingInput::Text(item.clone()))
                    .collect(),
            ),
            parameters: EmbeddingParams::default(),
        };

        let inputs = MaybeBatched::Batched(
            inputs
                .iter()
                .map(|item| EmbeddingInput::Text(item.clone()))
                .collect(),
        );

        assert_eq!(request.inputs, inputs)
    }

    #[test]
    fn test_embedding_params_creation() {
        let params = EmbeddingParams { normalize: true };
        assert!(params.normalize);

        let params = EmbeddingParams { normalize: false };
        assert!(!params.normalize);
    }

    #[test]
    fn test_embedding_params_clone() {
        let params = EmbeddingParams { normalize: true };
        assert_eq!(params.normalize, params.normalize);
    }
}
