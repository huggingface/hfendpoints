import asyncio
import logging
from asyncio import AbstractEventLoop

from hfendpoints.hfendpoints.openai import AutomaticSpeechRecognitionEndpoint


# class WhisperHandler(Handler):
#
#     def __new__(cls, model_id_or_path: str, ):
#         return super().__new__(cls, model_id_or_path)
#
#     async def __call__(self, request):
#         print(f"call: {request}")


class WhisperEndpoint(AutomaticSpeechRecognitionEndpoint):
    pass


async def entrypoint(loop: AbstractEventLoop):
    endpoint = WhisperEndpoint()
    await loop.run_in_executor(None, endpoint.run, "0.0.0.0", 8000)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(entrypoint(loop))
