import asyncio
from asyncio import AbstractEventLoop

from hfendpoints.hfendpoints.openai import TranscriptionEndpoint


class VllmTranscriptionEndpoint(TranscriptionEndpoint):
    def __init__(self):
        super().__init__()


async def entrypoint(loop: AbstractEventLoop):
    endpoint = VllmTranscriptionEndpoint()
    print(endpoint, "task:", endpoint.description())

    loop.run_in_executor(None, endpoint.run, "0.0.0.0", 8000)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(entrypoint(loop))
