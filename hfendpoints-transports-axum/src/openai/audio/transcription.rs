use axum::body::Bytes;
use axum::extract::Multipart;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// One segment of the transcribed text and the corresponding details.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Transcription {
    /// The transcribed text.
    text: String,
}

/// Represents a verbose json transcription response returned by model, based on the provided input.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.delta")]
pub struct Delta {
    /// The text delta that was additionally transcribed.
    pub(crate) delta: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-delta-event#audio/transcript-text-delta-event-logprobs
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.done")]
pub struct Done {
    /// The text that was transcribed.
    pub(crate) text: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-done-event#audio/transcript-text-done-event-logprobs
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Serialize, ToSchema)]
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
#[derive(Debug, Copy, Clone, Deserialize, ToSchema)]
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
#[derive(Debug, Clone, Serialize, ToSchema)]
pub enum TranscriptionResponse {
    Json(Transcription),
    Text(String),
    VerboseJson(VerboseTranscription),
}

/// Transcribes audio into the input language.
#[derive(ToSchema)]
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
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    file: Bytes,
    content_type: Option<&'static str>,
    language: Option<String>,
    prompt: Option<String>,
    temperature: Option<f32>,
    response_format: ResponseFormat,
}

// impl TranscriptionRequest {
//     async fn try_from_multipart(multipart: &mut Multipart) -> Result<Self, MultipartError> {
//         let mut file = None;
//         let mut content_type = None;
//         let mut language = None;
//         let mut prompt = None;
//         let mut temperature = None;
//         let mut response_format = None;
//
//         while let Some(field) = multipart.next_field().await.unwrap() {
//             let name = field.name().unwrap().to_string();
//
//             match name.as_str() {
//                 "file" => {
//                     file = Some(field.bytes().await?);
//                     content_type = field.content_type();
//                 }
//                 "language" => language = Some(field.text().await?),
//                 "prompt" => prompt = Some(field.text().await?),
//                 "temperature" => temperature = Some(f32::from_str(&field.text()?.await)?),
//                 "response_format" => match field.text().await?.as_str() {
//                     "json" => response_format = Some(ResponseFormat::Json),
//                     "text" => response_format = Some(ResponseFormat::Text),
//                     "verbose_json" => response_format = Some(ResponseFormat::VerboseJson),
//                 },
//                 _ => {}
//             }
//         }
//
//         if file.is_none() {
//             return Err(());
//         }
//
//         if response_format.is_none() {
//             return Err(());
//         }
//
//         Ok(Self {
//             file: file.unwrap(),
//             content_type,
//             language,
//             prompt,
//             temperature,
//             response_format: response_format.unwrap(),
//         })
//     }
// }

#[utoipa::path(
    post,
    path = "audio/transcriptions",
    tag = "transcriptions",
    request_body(content = TranscriptionForm, content_type = "multipart/form-data"),
    responses(
        (status = OK, description = "Transcribes audio into the input language.", body = Transcription),
        (status = OK, description = "Transcribes audio into the input language.", body = VerboseTranscription),
        (status = OK, description = "Transcribes audio into the input language.", body = String)
    )
)]
pub async fn create_transcription(mut multipart: Multipart) -> impl IntoResponse {
    // let request = TranscriptionRequest::try_from_multipart(&mut multipart)?;

    StatusCode::OK
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
