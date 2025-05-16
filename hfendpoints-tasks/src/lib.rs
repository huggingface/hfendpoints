use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use utoipa::ToSchema;

#[cfg(feature = "python")]
use pyo3::{FromPyObject, IntoPyObject};

pub mod audio;
pub mod embedding;

/// Container enum representing either a single element `T` or a sequence whose elements are of type `T`
///
/// # Examples
/// ```
/// use hfendpoints_tasks::MaybeBatched;
/// let single = MaybeBatched::Single("My name is Morgan");
/// let batch = MaybeBatched::Batched(vec!["My name is Morgan", "I'm working at Hugging Face"]);
/// ```
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
#[derive(Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
#[derive(PartialEq)]
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
/// use hfendpoints_tasks::Usage;
/// let usage = Usage { prompt_tokens: 1024, total_tokens: 1084 };
/// assert_eq!(usage.prompt_tokens, 1024);
/// assert_eq!(usage.total_tokens, 1084);
/// assert!(usage.prompt_tokens <= usage.total_tokens);
/// ```
///
/// If the two values are the same, you can you `Usage::same` helper factory function
/// ```
/// use hfendpoints_tasks::Usage;
/// let usage = Usage::same(100);
/// assert_eq!(usage.prompt_tokens, usage.total_tokens);
/// assert_eq!(usage.total_tokens, 100);
/// ```
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Copy, Clone, Deserialize, Serialize, ToSchema)]
pub struct Usage {
    /// Number of tokens included in the prompt after the tokenization process
    pub prompt_tokens: usize,

    /// Total number of tokens after the generation process, including prompt tokens
    pub total_tokens: usize,
}

impl Default for Usage {
    #[inline]
    fn default() -> Self {
        Usage {
            prompt_tokens: 0,
            total_tokens: 0,
        }
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

impl Usage {
    /// Creates a new instance of the `Usage` struct with the specified number of
    /// prompt tokens and total tokens.
    ///
    /// # Arguments
    ///
    /// * `prompt_tokens` - A `usize` value representing the number of tokens used in the prompt.
    /// * `total_tokens` - A `usize` value representing the total number of tokens used.
    ///
    /// # Returns
    ///
    /// A new instance of the `Usage` struct initialized with the provided token counts.
    ///
    /// # Example
    /// ```
    /// use hfendpoints_tasks::Usage;
    ///
    /// let usage = Usage::new(100, 128);
    /// assert_eq!(usage.prompt_tokens, 100);
    /// assert_eq!(usage.total_tokens, 128);
    /// ```
    pub fn new(prompt_tokens: usize, total_tokens: usize) -> Self {
        Usage {
            prompt_tokens,
            total_tokens,
        }
    }

    /// Creates a new `Usage` instance with the same value for `prompt_tokens` and `total_tokens`.
    ///
    /// # Parameters
    /// - `num_tokens` (usize): The number of tokens to be assigned to both `prompt_tokens` and `total_tokens`.
    ///
    /// # Returns
    /// - `Self`: A new `Usage` instance with `prompt_tokens` and `total_tokens` set to `num_tokens`.
    ///
    /// # Inline
    /// This function is marked as `#[inline]` to suggest to the compiler that it might be beneficial to inline it for performance reasons.
    ///
    /// # Example
    /// ```
    /// use hfendpoints_tasks::Usage;
    ///
    /// let usage = Usage::same(100);
    /// assert_eq!(usage.prompt_tokens, 100);
    /// assert_eq!(usage.total_tokens, 100);
    /// ```
    #[inline]
    pub fn same(num_tokens: usize) -> Self {
        Usage {
            prompt_tokens: num_tokens,
            total_tokens: num_tokens,
        }
    }
}

/// Generic request representation for endpoints
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
#[derive(Deserialize, ToSchema)]
pub struct EndpointRequest<I, P>
where
    I: ToSchema,
    P: ToSchema,
{
    /// Main processing input to feed through the inference engine
    pub inputs: I,

    /// Contains all the parameters to tune the inference engine
    parameters: P,
}

/// Generic response representation for endpoints
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", derive(FromPyObject))]
#[derive(Clone, Serialize, ToSchema)]
pub struct EndpointResponse<O, U>
where
    O: ToSchema,
    U: ToSchema,
{
    /// Resulting output from the API call
    #[serde(flatten)]
    pub output: O,

    /// Provide optional information about the underlying resources usage for this API call
    pub usage: Option<U>,
}

impl<I, P> EndpointRequest<I, P>
where
    I: ToSchema,
    P: ToSchema,
{
    #[inline]
    pub fn new(inputs: I, parameters: P) -> Self {
        Self { inputs, parameters }
    }
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
