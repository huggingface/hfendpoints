use crate::openai::audio::AUDIO_TAG;
use crate::openai::{OpenAiResult, OpenAiRouterFactory};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart};
use axum::response::IntoResponse;
use axum::Json;
use hfendpoints_inference_engine::InferService;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

use crate::openai::error::OpenAiError::ValidationError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use tracing::log::info;

/// One segment of the transcribed text and the corresponding details.
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Segment {
    /// Unique identifier of the segment.
    id: u16,

    /// Start time of the segment in seconds.
    start: f32,

    /// End time of the segment in seconds.
    end: f32,

    /// Seek offset of the segment.
    seek: u16,

    /// Temperature parameter used for generating the segment.
    temperature: f32,

    /// Text content of the segment.
    text: String,

    /// Array of token IDs for the text content.
    tokens: Vec<u32>,

    /// Average logprob of the segment.
    /// If the value is lower than -1, consider the logprobs failed.
    avg_logprob: f32,

    /// Compression ratio of the segment.
    /// If the value is greater than 2.4, consider the compression failed.
    compression_ratio: f32,

    /// Probability of no speech in the segment.
    /// If the value is higher than 1.0 and the avg_logprob is below -1, consider this segment silent.
    no_speech_prob: f32,
}

/// Represents a transcription response returned by model, based on the provided input.
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Transcription {
    /// The transcribed text.
    text: String,
}

/// Represents a verbose json transcription response returned by model, based on the provided input.
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct VerboseTranscription {
    /// The transcribed text.
    text: String,

    /// The duration of the input audio.
    duration: f32,

    /// The language of the input audio.
    language: String,

    /// Segments of the transcribed text and their corresponding details.
    segments: Vec<Segment>,
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.delta")]
pub struct Delta {
    /// The text delta that was additionally transcribed.
    pub(crate) delta: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-delta-event#audio/transcript-text-delta-event-logprobs
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.done")]
pub struct Done {
    /// The text that was transcribed.
    pub(crate) text: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-done-event#audio/transcript-text-done-event-logprobs
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(untagged)]
pub enum StreamEvent {
    /// Emitted when there is an additional text delta.
    /// This is also the first event emitted when the transcription starts.
    /// Only emitted when you create a transcription with the Stream parameter set to true.
    Delta(Delta),

    /// Emitted when the transcription is complete.
    /// Contains the complete transcription text.
    /// Only emitted when you create a transcription with the Stream parameter set to true.
    Done(Done),
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Copy, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    Json,
    Text,
    VerboseJson,
}

impl Default for ResponseFormat {
    #[inline]
    fn default() -> Self {
        ResponseFormat::Json
    }
}

/// The transcription object, a verbose transcription object or a stream of transcript events.
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub enum TranscriptionResponse {
    Json(Transcription),
    Text(String),
    VerboseJson(VerboseTranscription),
}

/// Transcribes audio into the input language.
#[derive(ToSchema)]
#[cfg_attr(debug_assertions, derive(Debug))]
struct TranscriptionForm {
    /// The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
    #[schema(format = Binary)]
    file: String,

    /// The language of the input audio.
    /// Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.
    language: Option<String>,

    /// Not used, here for compatibility purpose with OpenAI Platform
    model: Option<String>,

    /// An optional text to guide the model's style or continue a previous audio segment.
    /// The prompt should match the audio language.
    prompt: Option<String>,

    /// The sampling temperature, between 0 and 1.
    /// Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
    temperature: Option<f32>,

    /// The format of the output, in one of these options: json, text, verbose_json.
    response_format: Option<ResponseFormat>,
}

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone)]
pub struct TranscriptionRequest {
    pub file: Bytes,
    pub content_type: String,
    pub language: String,
    pub prompt: Option<String>,
    pub temperature: f32,
    pub response_format: ResponseFormat,
}

impl TranscriptionRequest {
    fn validate(
        file: Option<Bytes>,
        content_type: String,
        language: Option<String>,
        prompt: Option<String>,
        temperature: Option<f32>,
        response_format: Option<String>,
    ) -> OpenAiResult<Self> {
        let file = match file {
            Some(file) => Ok(file),
            None => Err(ValidationError(
                "Required parameter 'file' was not provided".to_string(),
            )),
        }?;

        let response_format = response_format.unwrap_or(String::from("json"));
        let response_format = match response_format.as_str() {
            "json" => Ok(ResponseFormat::Json),
            "verbose_json" => Ok(ResponseFormat::VerboseJson),
            "text" => Ok(ResponseFormat::Text),
            _ => Err(ValidationError(format!(
                "Unknown response_format: {response_format}. Possible values are: 'json', 'verbose_json', 'text'."
            ))),
        }?;

        let language = language.unwrap_or(String::from("en"));
        let temperature = temperature.unwrap_or(0.0);

        Ok(Self {
            file,
            content_type,
            language,
            prompt,
            temperature,
            response_format,
        })
    }

    async fn try_from_multipart(mut multipart: Multipart) -> OpenAiResult<Self> {
        let mut file: OpenAiResult<Option<Bytes>> = Ok(None);
        let mut content_type: Option<String> = None;
        let mut language: OpenAiResult<Option<String>> = Ok(None);
        let mut prompt: OpenAiResult<Option<String>> = Ok(None);
        let mut temperature: OpenAiResult<Option<f32>> = Ok(None);
        let mut response_format: OpenAiResult<Option<String>> = Ok(None);

        while let Some(field) = multipart.next_field().await? {
            let name = field.name().unwrap().to_string();
            match name.as_str() {
                "file" => {
                    content_type = Some(field.content_type().unwrap_or("unknown").to_string());
                    file = Ok(Some(field.bytes().await?));
                }
                "language" => language = Ok(Some(field.text().await?.to_string())),
                "prompt" => prompt = Ok(Some(field.text().await?.to_string())),
                "temperature" => temperature = Ok(Some(f32::from_str(&field.text().await?)?)),
                "response_format" => response_format = Ok(Some(field.text().await?.to_string())),
                _ => return Err(ValidationError(format!("Unknown field: {name}"))),
            }
        }

        Self::validate(
            file?,
            content_type.unwrap(),
            language?,
            prompt?,
            temperature?,
            response_format?,
        )
    }
}

#[utoipa::path(
    post,
    path = "/audio/transcriptions",
    tag = AUDIO_TAG,
    request_body(content = TranscriptionForm, content_type = "multipart/form-data"),
    responses(
        (status = OK, description = "Transcribes audio into the input language.", body = TranscriptionResponse),
    )
)]
pub async fn transcribe(multipart: Multipart) -> OpenAiResult<Json<&'static str>> {
    let request = TranscriptionRequest::try_from_multipart(multipart).await?;
    info!(
        "Received audio file {} ({} kB)",
        &request.content_type,
        request.file.len() / 1024
    );
    Ok(Json::from("Hello World"))
}

/// Helper factory to build
/// [OpenAi Platform compatible Transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription)
pub struct TranscriptionEndpointFactory;
impl OpenAiRouterFactory for TranscriptionEndpointFactory {
    fn description() -> &'static str {
        "Transcribes audio into the input language."
    }

    fn routes() -> OpenApiRouter {
        OpenApiRouter::new()
            .routes(routes!(transcribe))
            .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
    }
}

#[cfg(test)]
mod tests {
    use crate::openai::audio::transcription::{Delta, Done, StreamEvent};

    #[test]
    fn serialize_stream_event_delta() {
        let delta = StreamEvent::Delta(Delta {
            delta: String::from("Hello world"),
        });
        let delta_json =
            serde_json::to_string(&delta).expect("Failed to serialize StreamEvent::Delta");

        assert_eq!(
            &delta_json,
            r#"{"type":"transcript.text.delta","delta":"Hello world"}"#
        );
    }

    #[test]
    fn serialize_stream_event_done() {
        let done = StreamEvent::Done(Done {
            text: String::from("Hello world"),
        });
        let done_json =
            serde_json::to_string(&done).expect("Failed to serialize StreamEvent::Done");

        assert_eq!(
            &done_json,
            r#"{"type":"transcript.text.done","text":"Hello world"}"#
        );
    }
}
