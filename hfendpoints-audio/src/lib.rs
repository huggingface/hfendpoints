mod io {
    // use std::io::Cursor;
    // use symphonia::core::audio::{RawSample, RawSampleBuffer, SignalSpec};
    // use symphonia::core::codecs::{CodecParameters, DecoderOptions};
    // use symphonia::core::errors::Result as SymphoniaResult;
    // use symphonia::core::formats::FormatOptions;
    // use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    // use symphonia::core::meta::MetadataOptions;
    // use symphonia::core::probe::Hint;
    // use symphonia::core::units::Time;
    // use symphonia::default::{get_codecs, get_probe};
    // use tracing::instrument;
    //
    // #[instrument(skip_all)]
    // pub fn load_audio<T: RawSample>(wave: &[u8]) -> SymphoniaResult<(RawSampleBuffer<T>, Time, CodecParameters)> {
    //     let codecs = get_codecs();
    //     let probe = get_probe();
    //
    //     let cursor = Cursor::new(wave);
    //     let stream = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());
    //
    //     // Detect audio format
    //     let hint = Hint::default();
    //     let mut guess = probe.format(&hint, stream, &FormatOptions::default(), &MetadataOptions::default())?;
    //
    //     // Allocate audio decoder for the target audio format
    //     let mut decoder = codecs.make(&guess.format, &DecoderOptions::default())?;
    //     let codec_params = decoder.codec_params();
    //
    //     // Decode until the end
    //     let duration = codec_params.time_base.unwrap().calc_time(codec_params.n_frames.unwrap());
    //     let mut raw_audio_buffer = RawSampleBuffer::<T>::new(
    //         duration.seconds + 1,
    //         SignalSpec::new(codec_params.sample_rate.unwrap(), codec_params.channels.unwrap()),
    //     );
    //     loop {
    //         let packet = guess.format.next_packet()?;
    //         let decoded = decoder.decode(&packet)?;
    //         decoded.convert::<T>(&mut raw_audio_buffer);
    //     }
    //
    //     Ok((raw_audio_buffer, duration, codec_params.clone()))
    // }

    #[cfg(feature = "python")]
    pub(crate) mod python {
        use hfendpoints_binding_python::ImportablePyModuleBuilder;
        use pyo3::prelude::*;
        use symphonia::core::audio::RawSampleBuffer;
        use symphonia::core::codecs::CodecParameters;
        use symphonia::core::units::Time;

        #[pyclass(name = "NativeAudioBuffer")]
        pub struct PyAudioBuffer {
            pcm: RawSampleBuffer<f32>,
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
                self.codec.sample_rate.unwrap_or(0)
            }

            #[getter]
            fn channels(&self) -> usize {
                self.codec.channels.map(|channels| channels.count()).unwrap_or(0)
            }

            fn resample(&mut self) {}
        }

        // #[pyfunction(name = "load_audio_to_pcm")]
        // fn py_load_audio(content: &[u8]) -> PyResult<PyAudioBuffer> {
        //     let (pcm, duration, codec) = load_audio::<f32>(content)?;
        //     Ok(PyAudioBuffer { pcm, duration, codec })
        // }

        pub fn bind<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
            let module = ImportablePyModuleBuilder::new(py, name)?
                .add_class::<PyAudioBuffer>()?.finish();

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