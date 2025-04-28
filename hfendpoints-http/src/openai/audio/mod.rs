pub(crate) mod transcription;

pub const AUDIO_TAG: &str = "Audio";
pub const AUDIO_DESC: &str = "Learn how to turn audio into text or text into audio.";

#[cfg(feature = "python")]
pub(crate) mod python {
    use crate::openai::audio::transcription::python::TranscriptionResponseKind;
    use crate::openai::audio::transcription::{
        Segment, Transcription, TranscriptionRequest, TranscriptionResponse, VerboseTranscription,
    };
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::*;

    mod transcriptions {
        use crate::openai::audio::transcription::{
            TranscriptionRequest, TranscriptionResponse, TranscriptionRouter,
        };
        use crate::python::{impl_pyendpoint, impl_pyhandler};

        impl_pyhandler!(TranscriptionRequest, TranscriptionResponse);
        impl_pyendpoint!(
            "AutomaticSpeechRecognitionEndpoint",
            PyAutomaticSpeechRecognitionEndpoint,
            PyHandler,
            TranscriptionRouter
        );
    }

    /// Bind hfendpoints.http.audio submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            // transcription
            .add_class::<Segment>()?
            .add_class::<Transcription>()?
            .add_class::<VerboseTranscription>()?
            .add_class::<TranscriptionRequest>()?
            .add_class::<TranscriptionResponse>()?
            .add_class::<TranscriptionResponseKind>()?
            .add_class::<transcriptions::PyAutomaticSpeechRecognitionEndpoint>()?
            .finish();

        Ok(module)
    }
}
