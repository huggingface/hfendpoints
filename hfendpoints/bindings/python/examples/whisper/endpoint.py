import logging

from hfendpoints import Handler
from hfendpoints.openai.audio import AutomaticSpeechRecognitionEndpoint, TranscriptionRequest, TranscriptionResponse, \
    Segment, Transcription, VerboseTranscription


class WhisperHandler(Handler[TranscriptionRequest, TranscriptionResponse]):

    def __init__(self, model_id_or_path: str):
        super().__init__(model_id_or_path)

    def __call__(self, request: TranscriptionRequest, ctx) -> TranscriptionResponse:
        print(f"[Python] handler call: {request}")
        return TranscriptionResponse.text("Done from Python")


def entrypoint():
    endpoint = AutomaticSpeechRecognitionEndpoint(WhisperHandler("/repository"))
    endpoint.run("0.0.0.0", 8000)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    #
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.run_until_complete(entrypoint(loop))
    entrypoint()
