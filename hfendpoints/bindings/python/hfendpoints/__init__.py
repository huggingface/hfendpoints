from typing import Protocol, TypeVar

from hfendpoints import _hfendpoints

Request = TypeVar("Request", infer_variance=True)
Response = TypeVar("Response", infer_variance=True)


class Handler(Protocol[Request, Response]):

    def __init__(self, model_id_or_path: str):
        ...

    def __post_init__(self, context):
        ...

    def __call__(self, request: Request, context) -> Response:
        ...
