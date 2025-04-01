import asyncio
import logging
from asyncio import AbstractEventLoop

from hfendpoints import Handler
from hfendpoints.openai.audio import AutomaticSpeechRecognitionEndpoint


class WhisperHandler(Handler[object, object]):

    def __init__(self, model_id_or_path: str):
        super().__init__(model_id_or_path)
        print(f"New handler: {model_id_or_path}")

    def __post_init__(self, context):
        pass

    async def __call__(self, request: object, ctx) -> object:
        print(f"call: {request}, context: {ctx}")


async def entrypoint(loop: AbstractEventLoop):
    endpoint = AutomaticSpeechRecognitionEndpoint(WhisperHandler("/repository"))
    await loop.run_in_executor(None, endpoint.run, "0.0.0.0", 8000)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(entrypoint(loop))
