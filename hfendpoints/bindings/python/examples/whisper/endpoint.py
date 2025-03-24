from hfendpoints.hfendpoints.openai import TranscriptionEndpoint


class VllmTranscriptionEndpoint(TranscriptionEndpoint):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    endpoint = VllmTranscriptionEndpoint()
    print(endpoint, "task:", endpoint.description())
