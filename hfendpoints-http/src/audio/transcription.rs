use crate::audio::AUDIO_TAG;
use crate::context::Context;
use crate::headers::RequestId;
use crate::{HttpError, HttpResult, RequestWithContext};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum_extra::TypedHeader;
use hfendpoints_core::{EndpointContext, EndpointResult};
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use tokio::sync::mpsc::UnboundedSender;
use tracing::instrument;
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// One segment of the transcribed text and the corresponding details.
#[cfg_attr(feature = "python", pyclass(frozen))]
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

#[derive(Default)]
pub struct SegmentBuilder {
    id: Option<u16>,
    start: Option<f32>,
    end: Option<f32>,
    seek: Option<u16>,
    temperature: Option<f32>,
    text: Option<String>,
    tokens: Option<Vec<u32>>,
    avg_logprob: Option<f32>,
    compression_ratio: Option<f32>,
    no_speech_prob: Option<f32>,
}

impl SegmentBuilder {
    pub fn id(mut self, id: u16) -> Self {
        self.id = Some(id);
        self
    }

    pub fn start(mut self, start: f32) -> Self {
        self.start = Some(start);
        self
    }

    pub fn end(mut self, end: f32) -> Self {
        self.end = Some(end);
        self
    }

    pub fn seek(mut self, seek: u16) -> Self {
        self.seek = Some(seek);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }

    pub fn tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    pub fn avg_logprob(mut self, avg_logprob: f32) -> Self {
        self.avg_logprob = Some(avg_logprob);
        self
    }

    pub fn compression_ratio(mut self, compression_ratio: f32) -> Self {
        self.compression_ratio = Some(compression_ratio);
        self
    }

    pub fn no_speech_prob(mut self, no_speech_prob: f32) -> Self {
        self.no_speech_prob = Some(no_speech_prob);
        self
    }

    pub fn build(self) -> HttpResult<Segment> {
        Ok(Segment {
            id: self.id.ok_or(HttpError::Validation(String::from(
                "Segment::id is not set",
            )))?,
            start: self.start.ok_or(HttpError::Validation(String::from(
                "Segment::start is not set",
            )))?,
            end: self.end.ok_or(HttpError::Validation(String::from(
                "Segment::end is not set",
            )))?,
            seek: self.seek.unwrap_or(0),
            temperature: self.temperature.ok_or(HttpError::Validation(String::from(
                "Segment::temperature is not set",
            )))?,
            text: self.text.ok_or(HttpError::Validation(String::from(
                "Segment::text is not set",
            )))?,
            tokens: self.tokens.ok_or(HttpError::Validation(String::from(
                "Segment::tokens is not set",
            )))?,
            avg_logprob: self.avg_logprob.unwrap_or(0.0),
            compression_ratio: self.compression_ratio.unwrap_or(0.0),
            no_speech_prob: self.no_speech_prob.unwrap_or(0.0),
        })
    }
}

impl Segment {
    pub fn builder() -> SegmentBuilder {
        SegmentBuilder::default()
    }
}

/// Represents a transcription response returned by model, based on the provided input.
#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub struct Transcription {
    /// The transcribed text.
    text: String,
}

/// Represents a verbose json transcription response returned by model, based on the provided input.
#[cfg_attr(feature = "python", pyclass(frozen))]
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

#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.delta")]
pub struct Delta {
    /// The text delta that was additionally transcribed.
    pub(crate) delta: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-delta-event#audio/transcript-text-delta-event-logprobs
}

#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename = "transcript.text.done")]
pub struct Done {
    /// The text that was transcribed.
    pub(crate) text: String,
    // TODO: logprobs -> https://platform.openai.com/docs/api-reference/audio/transcript-text-done-event#audio/transcript-text-done-event-logprobs
}

#[cfg_attr(feature = "python", pyclass(frozen))]
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

#[cfg_attr(feature = "python", pyclass(frozen))]
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
#[cfg_attr(feature = "python", pyclass(frozen))]
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Serialize, ToSchema)]
pub enum TranscriptionResponse {
    Json(Transcription),
    Text(String),
    VerboseJson(VerboseTranscription),
}

impl IntoResponse for TranscriptionResponse {
    fn into_response(self) -> Response {
        match self {
            TranscriptionResponse::Json(transcription) => Json::from(transcription).into_response(),
            TranscriptionResponse::Text(text) => text.into_response(),
            TranscriptionResponse::VerboseJson(transcription) => {
                Json::from(transcription).into_response()
            }
        }
    }
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

#[cfg_attr(feature = "python", pyclass(frozen))]
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
    #[instrument(skip_all)]
    fn validate(
        file: Option<Bytes>,
        content_type: String,
        language: Option<String>,
        prompt: Option<String>,
        temperature: Option<f32>,
        response_format: Option<String>,
    ) -> HttpResult<Self> {
        let file = match file {
            Some(file) => Ok(file),
            None => Err(HttpError::Validation(
                "Required parameter 'file' was not provided".to_string(),
            )),
        }?;

        let response_format = response_format.unwrap_or(String::from("json"));
        let response_format = match response_format.as_str() {
            "json" => Ok(ResponseFormat::Json),
            "verbose_json" => Ok(ResponseFormat::VerboseJson),
            "text" => Ok(ResponseFormat::Text),
            _ => Err(HttpError::Validation(format!(
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

    #[instrument(skip_all)]
    async fn try_from_multipart(mut multipart: Multipart) -> HttpResult<Self> {
        let mut file: HttpResult<Option<Bytes>> = Ok(None);
        let mut content_type: Option<String> = None;
        let mut language: HttpResult<Option<String>> = Ok(None);
        let mut prompt: HttpResult<Option<String>> = Ok(None);
        let mut temperature: HttpResult<Option<f32>> = Ok(None);
        let mut response_format: HttpResult<Option<String>> = Ok(None);

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
                _ => return Err(HttpError::Validation(format!("Unknown field: {name}"))),
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

type TranscriptionRequestWithContext = RequestWithContext<TranscriptionRequest>;

#[utoipa::path(
    post,
    path = "/audio/transcriptions",
    tag = AUDIO_TAG,
    request_body(content = TranscriptionForm, content_type = "multipart/form-data"),
    responses(
        (status = OK, description = "Transcribes audio into the input language.", body = TranscriptionResponse),
    )
)]
#[instrument(skip(state, multipart))]
pub async fn transcribe(
    State(state): State<EndpointContext<TranscriptionRequestWithContext, TranscriptionResponse>>,
    request_id: TypedHeader<RequestId>,
    multipart: Multipart,
) -> HttpResult<TranscriptionResponse> {
    // Decode request
    let request = TranscriptionRequest::try_from_multipart(multipart).await?;

    // Create request context
    let ctx = Context::new(request_id.0);

    // Ask for the inference thread to handle it and wait for answers
    let mut egress = state.schedule((request, ctx));
    if let Some(response) = egress.recv().await {
        Ok(response?)
    } else {
        Err(HttpError::NoResponse)
    }
}

/// Helper factory to build
/// [OpenAi Platform compatible Transcription endpoint](https://platform.openai.com/docs/api-reference/audio/createTranscription)
#[derive(Clone)]
pub struct TranscriptionRouter(
    pub  UnboundedSender<(
        TranscriptionRequestWithContext,
        UnboundedSender<EndpointResult<TranscriptionResponse>>,
    )>,
);

impl From<TranscriptionRouter> for OpenApiRouter {
    fn from(value: TranscriptionRouter) -> Self {
        OpenApiRouter::new()
            .routes(routes!(transcribe))
            .with_state(EndpointContext::new(value.0))
            .layer(DefaultBodyLimit::max(200 * 1024 * 1024)) // 200Mb as OpenAI
    }
}

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::audio::transcription::{
        ResponseFormat, Segment, Transcription, TranscriptionRequest, TranscriptionResponse,
        VerboseTranscription,
    };
    use hfendpoints_binding_python::fill_view_from_readonly_data;
    use pyo3::ffi::Py_buffer;
    use pyo3::prelude::*;
    use std::ffi::CString;
    use tracing::{debug, instrument};

    #[pyclass(frozen, eq, eq_int)]
    #[derive(Eq, PartialEq)]
    pub enum TranscriptionResponseKind {
        #[pyo3(name = "TEXT")]
        Text = 1,

        #[pyo3(name = "JSON")]
        Json = 2,

        #[pyo3(name = "VERBOSE_JSON")]
        VerboseJson = 3,
    }

    #[pymethods]
    impl Segment {
        #[new]
        pub fn new(
            id: u16,
            start: f32,
            end: f32,
            seek: u16,
            temperature: f32,
            text: String,
            tokens: Vec<u32>,
            avg_logprob: f32,
            compression_ratio: f32,
            no_speech_prob: f32,
        ) -> PyResult<Self> {
            Ok(Self {
                id,
                start,
                end,
                seek,
                temperature,
                text,
                tokens,
                avg_logprob,
                compression_ratio,
                no_speech_prob,
            })
        }
    }

    #[pymethods]
    impl Transcription {
        #[new]
        pub fn new(text: String) -> Self {
            Self { text }
        }
    }

    #[pymethods]
    impl VerboseTranscription {
        #[new]
        pub fn new(text: String, duration: f32, language: String, segments: Vec<Segment>) -> Self {
            Self {
                text,
                duration,
                language,
                segments,
            }
        }
    }

    #[pymethods]
    impl TranscriptionRequest {
        #[instrument(skip(slf, buffer))]
        pub unsafe fn __getbuffer__(
            slf: Bound<'_, Self>,
            buffer: *mut Py_buffer,
            flags: i32,
        ) -> PyResult<()> {
            debug!("Acquiring a memoryview over audio data (flags={})", flags);
            unsafe {
                fill_view_from_readonly_data(buffer, flags, &slf.borrow().file, slf.into_any())
            }
        }

        #[instrument(skip_all)]
        pub unsafe fn __releasebuffer__(&self, buffer: *mut Py_buffer) {
            debug!("Releasing Python memoryview");
            // Release memory held by the format string
            drop(unsafe { CString::from_raw((*buffer).format) });
        }

        #[getter]
        pub fn language(&self) -> &str {
            &self.language
        }

        #[getter]
        pub fn prompt(&self) -> &Option<String> {
            &self.prompt
        }

        #[getter]
        pub fn temperature(&self) -> f32 {
            self.temperature
        }

        #[getter]
        pub fn response_kind(&self) -> PyResult<TranscriptionResponseKind> {
            match self.response_format {
                ResponseFormat::Json => Ok(TranscriptionResponseKind::Json),
                ResponseFormat::Text => Ok(TranscriptionResponseKind::Text),
                ResponseFormat::VerboseJson => Ok(TranscriptionResponseKind::VerboseJson),
            }
        }
    }

    #[pymethods]
    impl TranscriptionResponse {
        #[staticmethod]
        fn text(content: String) -> Self {
            Self::Text(content)
        }

        #[staticmethod]
        fn json(content: String) -> Self {
            Self::Json(Transcription { text: content })
        }

        #[staticmethod]
        fn verbose(transcription: VerboseTranscription) -> Self {
            Self::VerboseJson(transcription)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::audio::transcription::{Delta, Done, Segment, StreamEvent};

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

    #[test]
    fn segment_builder_all_field_set() {
        if let Ok(segment) = Segment::builder()
            .id(1)
            .start(2.2)
            .end(3.8)
            .seek(7)
            .temperature(1.0)
            .text(String::from("Hello"))
            .tokens(vec![1, 2, 3])
            .avg_logprob(2.71)
            .compression_ratio(1.2)
            .no_speech_prob(0.1)
            .build()
        {
            assert_eq!(segment.id, 1);
            assert_eq!(segment.start, 2.2);
            assert_eq!(segment.end, 3.8);
            assert_eq!(segment.seek, 7);
            assert_eq!(segment.temperature, 1.0);
            assert_eq!(segment.text, String::from("Hello"));
            assert_eq!(segment.tokens, vec![1, 2, 3]);
            assert_eq!(segment.avg_logprob, 2.71);
            assert_eq!(segment.compression_ratio, 1.2);
            assert_eq!(segment.no_speech_prob, 0.1);
        } else {
            panic!("Failed to create segment");
        }
    }

    #[test]
    fn segment_builder_with_default_fields() {
        if let Ok(segment) = Segment::builder()
            .id(1)
            .start(2.2)
            .end(3.8)
            .temperature(1.0)
            .text(String::from("Hello"))
            .tokens(vec![1, 2, 3])
            .build()
        {
            assert_eq!(segment.id, 1);
            assert_eq!(segment.start, 2.2);
            assert_eq!(segment.end, 3.8);
            assert_eq!(segment.seek, 0);
            assert_eq!(segment.temperature, 1.0);
            assert_eq!(segment.text, String::from("Hello"));
            assert_eq!(segment.tokens, vec![1, 2, 3]);
            assert_eq!(segment.avg_logprob, 0.0);
            assert_eq!(segment.compression_ratio, 0.0);
            assert_eq!(segment.no_speech_prob, 0.0);
        } else {
            panic!("Failed to create segment");
        }
    }
}
