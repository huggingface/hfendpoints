from enum import Enum
from typing import List, Union


class EncodingFormat(Enum):
    FLOAT = 0
    BASE64 = 1


class EmbeddingRequest:

    def encoding_format(self) -> EncodingFormat: ...

    @property
    def is_batched(self) -> bool: ...

    @property
    def input(self) -> Union[str, List[str]]: ...
