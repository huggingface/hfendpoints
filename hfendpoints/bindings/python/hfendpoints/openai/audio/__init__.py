from typing import List

from ..._hfendpoints.openai.audio import AutomaticSpeechRecognitionEndpoint, Segment, Transcription, \
    VerboseTranscription, TranscriptionRequest, TranscriptionResponse, TranscriptionResponseKind


class SegmentBuilder:
    def __init__(self):
        self._id = None
        self._start = 0.0
        self._end = 0.0
        self._seek = 0
        self._temperature = 0.0
        self._text = None
        self._tokens = None
        self._avg_lobprob = 0.0
        self._compression_ratio = 0.0
        self._no_speech_prob = 0.0

    def build(self, ) -> Segment:
        if self._id is None:
            raise ValueError("Segment's id cannot be None")

        if self._text is None:
            raise ValueError("Segment's text cannot be None")

        if self._tokens is None:
            raise ValueError("Segment's tokens cannot be None")

        return Segment(
            id=self._id,
            start=self._start,
            end=self._end,
            seek=self._seek,
            temperature=self._temperature,
            text=self._text,
            tokens=self._tokens,
            avg_logprob=self._avg_lobprob,
            compression_ratio=self._compression_ratio,
            no_speech_prob=self._no_speech_prob
        )

    def id(self, id: int) -> "SegmentBuilder":
        self._id = id
        return self

    def start(self, start: float) -> "SegmentBuilder":
        self._start = start
        return self

    def end(self, end: float) -> "SegmentBuilder":
        self._end = end
        return self

    def seek(self, seek: int) -> "SegmentBuilder":
        self._seek = seek
        return self

    def temperature(self, temperature: float) -> "SegmentBuilder":
        self._temperature = temperature
        return self

    def text(self, text: str) -> "SegmentBuilder":
        self._text = text
        return self

    def tokens(self, tokens: List[int]) -> "SegmentBuilder":
        self._tokens = tokens
        return self

    def avg_logprob(self, avg_logprob: float) -> "SegmentBuilder":
        self._avg_lobprob = avg_logprob
        return self

    def compression_ratio(self, compression_ratio: float) -> "SegmentBuilder":
        self._compression_ratio = compression_ratio
        return self

    def no_speech_prob(self, no_speech_prob: float) -> "SegmentBuilder":
        self._no_speech_prob = no_speech_prob
        return self
