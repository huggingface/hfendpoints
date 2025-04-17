from typing import List

from . import HfEndpointException


class UnsupportedModelArchitecture(HfEndpointException):
    """
    Raised when a model architecture is not supported by the current endpoint
    """

    def __init__(self, archs: List[str], supported_archs: List[str]):
        super().__init__(
            f"This endpoint supports the following architectures {supported_archs} "
            f"but got these: {archs}"
        )
