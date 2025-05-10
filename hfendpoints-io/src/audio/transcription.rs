use crate::{EndpointRequest, EndpointResponse, Usage};
use hfendpoints_core::Handler;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(name = "Segment", frozen))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Segment {
    id: usize,
}

/// Describe all the parameters to tune the underlying transcription process
#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(feature = "python", pyclass(name = "TranscriptionParams", frozen))]
#[derive(Deserialize, ToSchema)]
pub struct TranscriptionParams {
    /// An optional text to guide the model's style or continue a previous audio segment.
    ///      
    /// The prompt should match the audio language.
    prompt: Option<String>,

    /// The language of the input audio.
    ///
    /// Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.
    language: Option<String>,

    /// The sampling temperature, between 0 and 1.
    ///
    /// Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    temperature: Option<f32>,

    /// The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Option<usize>,

    /// If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_p: Option<f32>,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Serialize, ToSchema)]
pub struct DetailedTranscription {
    text: String,
    segments: Vec<Segment>,
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Serialize, ToSchema)]
pub enum Transcription {
    Text(String),
    Detailed(DetailedTranscription),
}

/// Endpoint request specification for Transcription endpoints
pub type TranscriptionRequest = EndpointRequest<String, TranscriptionParams>;

/// Endpoint response specification for Transcription endpoints
pub type TranscriptionResponse = EndpointResponse<Transcription, Usage>;

/// Helper trait to implement `Handler` specification for Transcription endpoints
pub trait TranscriptionHandler:
    Handler<Request = Self::TypedRequest, Response = Self::TypedResponse>
{
    type TypedRequest: Into<TranscriptionRequest>;
    type TypedResponse: From<TranscriptionResponse>;
}

impl Default for TranscriptionParams {
    #[inline]
    fn default() -> Self {
        Self {
            prompt: None,
            language: Some(String::from("en")),
            temperature: Some(0.0),
            top_k: None,
            top_p: Some(1.0),
        }
    }
}

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::audio::transcription::{
        DetailedTranscription, Segment, Transcription, TranscriptionParams, TranscriptionRequest,
        TranscriptionResponse,
    };
    use crate::Usage;
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;
    use pyo3::{Bound, PyResult, Python};

    #[pyclass(name = "TranscriptionRequest", frozen)]
    pub struct PyTranscriptionRequest(TranscriptionRequest);

    #[pyclass(name = "TranscriptionResponse", frozen)]
    pub struct PyTranscriptionResponse(TranscriptionResponse);

    #[pymethods]
    impl PyTranscriptionResponse {
        #[new]
        #[pyo3(signature = (text, /, segments=None, prompt_tokens=None, total_tokens=None))]
        fn new(
            text: String,
            segments: Option<Vec<Segment>>,
            prompt_tokens: Option<usize>,
            total_tokens: Option<usize>,
        ) -> Self {
            Self {
                0: TranscriptionResponse {
                    output: match segments {
                        None => Transcription::Text(text),
                        Some(segments) => {
                            Transcription::Detailed(DetailedTranscription { text, segments })
                        }
                    },
                    usage: match (prompt_tokens, total_tokens) {
                        (None, None) => None,
                        (Some(prompt), None) => Some(Usage::same(prompt)),
                        (None, Some(total)) => Some(Usage::new(0, total)),
                        (Some(prompt), Some(total)) => Some(Usage::new(prompt, total)),
                    },
                },
            }
        }

        fn __repr__(&self) -> String {
            let text = match &self.0.output {
                Transcription::Text(text) => &text,
                Transcription::Detailed(detailed) => &detailed.text,
            };

            match &self.0.usage {
                None => format!("TranscriptionResponse(text={text})"),
                Some(usage) => format!("TranscriptionResponse(text={text}, usage={usage})"),
            }
        }
    }

    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_class::<Segment>()?
            .add_class::<TranscriptionParams>()?
            .add_class::<PyTranscriptionRequest>()?
            .add_class::<PyTranscriptionResponse>()?
            .finish();

        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_params_default() {
        let params = TranscriptionParams::default();
        assert_eq!(params.prompt, None);
        assert_eq!(params.language, Some(String::from("en")));
        assert_eq!(params.temperature, Some(0.0));
        assert_eq!(params.top_k, None);
        assert_eq!(params.top_p, Some(1.0));
    }

    #[test]
    fn test_transcription_params_custom() {
        let params = TranscriptionParams {
            prompt: Some(String::from("test prompt")),
            language: Some(String::from("fr")),
            temperature: Some(0.8),
            top_k: Some(50),
            top_p: Some(0.9),
        };

        assert_eq!(params.prompt, Some(String::from("test prompt")));
        assert_eq!(params.language, Some(String::from("fr")));
        assert_eq!(params.temperature, Some(0.8));
        assert_eq!(params.top_k, Some(50));
        assert_eq!(params.top_p, Some(0.9));
    }
}
