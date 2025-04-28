use axum::extract::State;
use axum_extra::TypedHeader;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::error::OpenAiError;
use crate::headers::RequestId;
use crate::huggingface::HuggingFaceRequest;
use crate::openai::audio::AUDIO_TAG;
use crate::{Context, OpenAiResult};
use hfendpoints_core::{EndpointContext, Error};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use utoipa::ToSchema;

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Copy, PartialEq, Eq, Deserialize, ToSchema)]
pub enum AutomaticSpeechRecognitionEarlyStoppingEnum {
    True,
    False,
    Never,
}

/// Parametrization of the text generation process
#[allow(unused)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Copy, Deserialize, ToSchema)]
pub(crate) struct AutomaticSpeechRecognitionGenerationParams {
    /// Whether to use sampling instead of greedy decoding when generating new tokens.
    do_sample: Option<bool>,

    /// Controls the stopping condition for beam-based methods.
    early_stopping: Option<AutomaticSpeechRecognitionEarlyStoppingEnum>,

    /// If set to float strictly between 0 and 1, only tokens with a conditional probability
    /// greater than epsilon_cutoff will be sampled. In the paper, suggested values range from
    /// 3e-4 to 9e-4, depending on the size of the model.
    /// See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    epsilon_cutoff: Option<f32>,

    /// Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to
    /// float strictly between 0 and 1, a token is only considered if it is greater than either
    /// eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). The latter
    /// term is intuitively the expected next token probability, scaled by sqrt(eta_cutoff). In
    /// the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model.
    /// See [Truncation Sampling as Language Model Desmoothing](https://hf.co/papers/2210.15191) for more details.
    eta_cutoff: Option<f32>,

    /// The maximum length (in tokens) of the generated text, including the input.
    max_length: Option<usize>,

    /// The maximum number of tokens to generate. Takes precedence over max_length.
    max_new_tokens: Option<usize>,

    /// The minimum length (in tokens) of the generated text, including the input.
    min_length: Option<usize>,

    /// The minimum number of tokens to generate. Takes precedence over min_length.
    min_new_tokens: Option<usize>,

    /// Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.
    /// See [this paper](https://hf.co/papers/1610.02424) for more details.
    num_beam_groups: Option<u8>,

    /// Number of beams to use for beam search.
    num_beams: Option<u8>,

    /// The value balances the model confidence and the degeneration penalty in contrastive search decoding.
    penalty_alpha: Option<f32>,

    /// The value used to modulate the next token probabilities.
    temperature: Option<f32>,

    /// The number of highest probability vocabulary tokens to keep for top-k-filtering.
    top_k: Option<u16>,

    /// If set to float < 1, only the smallest set of most probable tokens with probabilities
    /// that add up to top_p or higher are kept for generation.
    top_p: Option<f32>,

    /// Local typicality measures how similar the conditional probability of predicting a target
    /// token next is to the expected conditional probability of predicting a random token next,
    /// given the partial text already generated.
    /// If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation.
    /// See [this paper](https://hf.co/papers/2202.00666) for more details.
    typical_p: Option<f32>,

    /// Whether the model should use the past last key/values attentions to speed up decoding
    use_cache: Option<bool>,
}

/// Additional inference parameters for Automatic Speech Recognition
#[allow(unused)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Copy, Deserialize, ToSchema)]
pub struct AutomaticSpeechRecognitionParams {
    /// Parametrization of the text generation process
    generation_params: Option<AutomaticSpeechRecognitionGenerationParams>,

    /// Whether to output corresponding timestamps with the generated text
    return_timestamps: bool,
}

/// Inputs for Automatic Speech Recognition inference
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct AutomaticSpeechRecognitionChunk {
    /// The input audio data as a base64-encoded string.
    /// If no `parameters` are provided, you can also provide the audio data as a raw bytes payload.
    text: String,

    /// The start and end timestamps corresponding with the text
    timestamps: Vec<f32>,
}

pub type AutomaticSpeechRecognitionRequest =
    HuggingFaceRequest<String, AutomaticSpeechRecognitionParams>;

/// Outputs of inference for the Automatic Speech Recognition task
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct AutomaticSpeechRecognitionResponse {
    /// A chunk of text identified by the model
    text: String,

    /// The start and end timestamps corresponding with the text
    chunks: Vec<AutomaticSpeechRecognitionChunk>,
}

#[utoipa::path(
    post,
    path = "/predict",
    tag = AUDIO_TAG,
    request_body(content = AutomaticSpeechRecognitionRequest, content_type = "application/json"),
    responses(
        (status = OK, description = "Transcribes audio into the input language.", body = AutomaticSpeechRecognitionResponse),
    )
)]
#[instrument(skip(state, request))]
pub async fn predict(
    State(state): State<
        EndpointContext<
            (AutomaticSpeechRecognitionRequest, Context),
            AutomaticSpeechRecognitionResponse,
        >,
    >,
    request_id: TypedHeader<RequestId>,
    request: AutomaticSpeechRecognitionRequest,
) -> OpenAiResult<AutomaticSpeechRecognitionResponse> {
    // Create request context
    let ctx = Context::new(request_id.0);

    // Ask for the inference thread to handle it and wait for answers
    let mut egress = state.schedule((request, ctx));
    if let Some(response) = egress.recv().await {
        Ok(response?)
    } else {
        Err(OpenAiError::NoResponse)
    }
}

/// Helper factory to build
/// [OpenAi Platform compatible Transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription)
#[derive(Clone)]
pub struct AutomaticSpeechRecognitionRouter(
    pub  UnboundedSender<(
        (AutomaticSpeechRecognitionRequest, Context),
        UnboundedSender<Result<AutomaticSpeechRecognitionResponse, Error>>,
    )>,
);
