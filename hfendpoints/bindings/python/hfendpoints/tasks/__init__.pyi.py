from typing import Self


class Usage:
    def __init__(self, prompt_tokens: int, total_tokens: int) -> "Self": ...
