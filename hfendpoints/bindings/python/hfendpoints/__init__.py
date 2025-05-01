from typing import Protocol, TypeVar, runtime_checkable

from packaging.version import Version
from hfendpoints._hfendpoints import __version__

from .config import EndpointConfig, ensure_supported_architectures

VERSION = Version(__version__)

Request = TypeVar("Request", infer_variance=True)
Response = TypeVar("Response", infer_variance=True)


@runtime_checkable
class Handler(Protocol[Request, Response]):

    def __init__(self, model_id_or_path: str): ...

    def __call__(self, request: Request, ctx) -> Response: ...
