use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use utoipa::ToSchema;

#[cfg(feature = "python")]
use pyo3::FromPyObject;

pub mod audio;
pub mod embedding;

/// Container enum representing either a single element `T` or a sequence whose elements are of type `T`
///
/// # Examples
/// ```
/// use hfendpoints_io::MaybeBatched;
/// let single = MaybeBatched::Single("My name is Morgan");
/// let batch = MaybeBatched::Batched(vec!["My name is Morgan", "I'm working at Hugging Face"]);
/// ```
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(FromPyObject))]
#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MaybeBatched<T> {
    /// Single element
    Single(T),

    /// Sequence of elements
    Batched(Vec<T>),
}

/// The `Usage` structure represents information about token usage during a text generation process.
///
/// # Attributes
///
/// * `prompt_tokens` -
///   The number of tokens included in the prompt after tokenization. This indicates the size of
///   the input prompt as measured in tokens.
///
/// * `total_tokens` -
///   The total number of tokens after the text generation process. This includes both the tokens
///   from the prompt and the tokens generated as output.
///
/// # Derives
///
/// * `Serialize` -
///   Enables the `Usage` struct to be serialized, typically for use in contexts like JSON APIs.
///
/// * `ToSchema` -
///   Allows the `Usage` struct to be used as a schema definition in OpenAPI (or similar) documentation.
///
/// # Optional
///
/// * When compiled in debug mode (with `debug_assertions` enabled), the `Debug` trait is
///   automatically derived for the struct, enabling debug-friendly formatting of its instances.
///
/// This struct is typically used to track and report token consumption in natural language processing tasks.
///
/// # Examples
///
/// ```
/// use hfendpoints_io::Usage;
/// let usage = Usage { prompt_tokens: 1024, total_tokens: 1084 };
/// assert_eq!(usage.prompt_tokens, 1024);
/// assert_eq!(usage.total_tokens, 1084);
/// assert!(usage.prompt_tokens <= usage.total_tokens);
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Serialize, ToSchema)]
pub struct Usage {
    /// Number of tokens included in the prompt after the tokenization process
    pub prompt_tokens: usize,

    /// Total number of tokens after the generation process, including prompt tokens
    pub total_tokens: usize,
}

impl Usage {
    #[inline]
    pub fn new(prompt_tokens: usize, total_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            total_tokens,
        }
    }

    #[inline]
    pub fn same(tokens: usize) -> Self {
        Self::new(tokens, tokens)
    }
}

impl Display for Usage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Usage(prompt_tokens={}, total_tokens={})",
            self.prompt_tokens, self.total_tokens
        )
    }
}

/// Generic request representation for endpoints
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Deserialize, ToSchema)]
pub struct EndpointRequest<I, P>
where
    I: ToSchema,
    P: ToSchema,
{
    /// Main processing input to feed through the inference engine
    inputs: I,

    /// Contains all the parameters to tune the inference engine
    parameters: P,
}

/// Generic response representation for endpoints
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Serialize, ToSchema)]
pub struct EndpointResponse<O, U>
where
    O: ToSchema,
    U: ToSchema,
{
    /// Resulting output from the API call
    #[serde(flatten)]
    output: O,

    /// Provide optional information about the underlying resources usage for this API call
    usage: Option<U>,
}

#[cfg(feature = "python")]
pub mod python {
    use crate::{audio, embedding};
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::PyModule;
    use pyo3::{Bound, PyResult, Python};

    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_submodule(&embedding::python::bind(py, &format!("{name}.embedding"))?)?
            .add_submodule(&audio::python::bind(py, &format!("{name}.audio"))?)?
            .finish();
        Ok(module)
    }
}
