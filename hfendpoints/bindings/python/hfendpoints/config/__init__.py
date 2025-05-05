import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..errors import UnsupportedModelArchitecture

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """
    Configuration object for an endpoint
    """

    # Flag indicating if the current endpoint runs in debug
    is_debug: bool

    # Interface the endpoint will be listening to incoming requests
    interface: str

    # Port on the interface the endpoint will be listening to incoming requests
    port: int

    # Model to use for this endpoint
    model_id: str

    @staticmethod
    def from_env() -> "EndpointConfig":
        """
        Parse the operating system environment variables to retrieve Inference Endpoints defined variables
        :return:
        """
        return EndpointConfig(
            interface=os.environ.get("INTERFACE", "0.0.0.0"),
            port=int(os.environ.get("PORT", 8000)),
            model_id=os.environ.get("HF_MODEL_ID", "unknown"),
            is_debug=os.environ.get("MODEL_ID", "/repository") != "/repository"
        )


def ensure_supported_architectures(
        config: "PretrainedConfig", supported_archs: Iterable[str]
):
    """
    Check whether the specified architectures for the provided `config` are supported according to the `supported_archs`
    :param config: `transformers.PretrainedConfig` object containing information about the underlying model architectures
    :param supported_archs: Iterable of strings representing the supported architectures by the endpoint
    :raises UnsupportedModelArchitecture if the union of config.architectures and supported_archs is empty
    """
    if config.architectures and not set(config.architectures).union(supported_archs):
        raise UnsupportedModelArchitecture(
            archs=config.architectures, supported_archs=list(supported_archs)
        )
