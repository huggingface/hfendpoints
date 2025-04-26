mod io {
    use std::io::{Error as IoError, ErrorKind as IoErrorKind};
    use symphonia::core::audio::conv::FromSample;
    use symphonia::core::audio::sample::{i24, u24, Sample};
    use symphonia::core::audio::{Audio, AudioBuffer};
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::codecs::CodecParameters;
    use symphonia::core::errors::Result as SymphoniaResult;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::{BufReader, MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::units::Time;
    use symphonia::default::{get_codecs, get_probe};
    use tracing::instrument;

    #[instrument(skip_all)]
    pub fn load_audio<T>(wave: &[u8]) -> SymphoniaResult<(Vec<f32>, Time, CodecParameters)>
    where
        T: Sample
            + FromSample<u8>
            + FromSample<u16>
            + FromSample<u24>
            + FromSample<u32>
            + FromSample<i8>
            + FromSample<i16>
            + FromSample<i24>
            + FromSample<i32>
            + FromSample<f32>
            + FromSample<f64>,
    {
        let codecs = get_codecs();
        let probe = get_probe();

        let raw_audio = Box::new(BufReader::new(wave));
        let stream = MediaSourceStream::new(raw_audio, MediaSourceStreamOptions::default());

        // Detect audio format
        let hint = Hint::default();
        let mut guess = probe.probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )?;

        // Allocate audio decoder for the target audio format
        let track = guess
            .default_track(TrackType::Audio)
            .ok_or(IoError::new(IoErrorKind::InvalidData, "Failed to decode audio as no track was discovered while skimming through the provided data."))?;

        let mut decoder = codecs.make_audio_decoder(
            &track.codec_params.as_ref().unwrap().audio().unwrap(),
            &AudioDecoderOptions::default(),
        )?;

        // let mut raw_audio_buffer = RawSampleBuffer::<T>::new(
        //     duration.seconds + 1,
        //
        //     SignalSpec::new(
        //         codec_params.sample_rate.unwrap(),
        //         codec_params.channels.unwrap(),
        //     ),
        // );

        track.

        loop {
            let packet = guess.format.next_packet()?;
            let decoded = decoder.decode(&packet)?;

            let mut converted = AudioBuffer::<T>::new(decoded.spec().clone(), decoded.capacity());

            decoded.copy_to(&mut converted);
            converted.copy_to_vec_interleaved(&mut out)
        }

        Ok((raw_audio_buffer, duration, codec_params.clone()))
    }

    #[cfg(feature = "python")]
    pub(crate) mod python {
        use crate::io::load_audio;
        use hfendpoints_binding_python::ImportablePyModuleBuilder;
        use pyo3::exceptions::PyIOError;
        use pyo3::prelude::*;
        use symphonia::core::codecs::CodecParameters;
        use symphonia::core::units::Time;

        #[pyclass(name = "NativeAudioBuffer")]
        pub struct PyAudioBuffer {
            pcm: Vec<f32>,
            duration: Time,
            codec: CodecParameters,
        }

        #[pymethods]
        impl PyAudioBuffer {
            #[getter]
            fn duration(&self) -> f64 {
                self.duration.seconds as f64 + self.duration.frac
            }

            #[getter]
            fn sample_rate(&self) -> u32 {
                self.codec
                    .audio()
                    .map(|audio| audio.sample_rate.unwrap_or(0))
                    .unwrap_or(0)
            }

            #[getter]
            fn channels(&self) -> usize {
                self.codec
                    .audio()
                    .map(|audio| {
                        audio
                            .channels
                            .as_ref()
                            .map(|channel| channel.count())
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            }

            fn resample(&mut self) {}
        }

        #[pyfunction(name = "load_audio_to_pcm")]
        fn py_load_audio(content: &[u8]) -> PyResult<PyAudioBuffer> {
            let (pcm, duration, codec) = load_audio::<f32>(content).map_err(|err| {
                PyIOError::new_err(format!("Failed to decode audio to PCM: {err}"))
            })?;
            Ok(PyAudioBuffer {
                pcm,
                duration,
                codec,
            })
        }

        pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
            let module = ImportablePyModuleBuilder::new(py, name)?
                .add_class::<PyAudioBuffer>()?
                .finish();

            // module.add_function(wrap_pyfunction!(py_load_audio)?)?;
            Ok(module)
        }
    }
}

#[cfg(feature = "python")]
pub mod python {
    use hfendpoints_binding_python::ImportablePyModuleBuilder;
    use pyo3::prelude::PyModule;
    use pyo3::{Bound, PyResult, Python};

    /// Bind hfendpoints.audio submodule into the exported Python wheel
    pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
        let module = ImportablePyModuleBuilder::new(py, name)?
            .defaults()?
            .add_submodule(&crate::io::python::bind(py, &format!("{name}.io"))?)?
            .finish();

        Ok(module)
    }
}
