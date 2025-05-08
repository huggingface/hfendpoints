from dataclasses import dataclass
from typing import Optional


class Segment:
    """

    """
    ...


@dataclass
class TranscriptionParams:
    """
    Describe all the parameters to tune the underlying transcription process
    """

    """
    An optional text to guide the model's style or continue a previous audio segment.
     
    The prompt should match the audio language.
    """
    prompt: Optional[str] = None

    """
    The language of the input audio. 
    
    Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.
    """
    language: str = "en"

    """
    The sampling temperature, between 0 and 1. 
    
    Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    """
    temperature: float = 0.0

    """
    The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """
    top_k: Optional[int] = None

    """
    If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    """
    top_p: Optional[float] = None


class TranscriptionRequest:
    """

    """

    @property
    def inputs(self) -> str:
        """
        The binary audio file-content to transcribe

        :return: `str`
        """
        ...

    @property
    def parameters(self) -> TranscriptionParams:
        """
        Access the parameters to tune the underlying transcription process

        :return: `TranscriptionParams`
        """
        ...


class TranscriptionResponse:
    """

    """

    def __init__(
            self,
            text: str,
            /,
            segments: Optional[Segment] = None,
            prompt_tokens: Optional[int] = None,
            total_tokens: Optional[int] = None
    ):
        """

        :param text:
        :param segments:
        :param prompt_tokens:
        :param total_tokens:
        """

        ...
