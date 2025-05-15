from dataclasses import dataclass
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self


@dataclass
class EmbeddingParams:
    pass


@dataclass
class EmbeddingRequest:
    """
    Embedding request representation for endpoints
    """

    inputs: Union[str, List[str]]

    parameters: EmbeddingParams


@dataclass
class EmbeddingResponse:
    """
    Embedding response representation for endpoints
    """

    def __init__(self, embeddings: Union[List[float], List[List[float]]], num_tokens: int) -> "Self":
        """
        Create an Embedding response from a Python allocated array of embeddings

        :param embeddings:
        :param num_tokens:
        """
        ...

    @staticmethod
    def from_numpy(embeddings: "np.ndarray", num_tokens: int) -> "Self":
        """
        Create an Embedding response from a Python numpy array of embeddings

        :param embeddings:
        :param num_tokens:
        :return:
        """
        ...
