from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from utils import logger
from utils.config import LLMConfig

RequestType = TypeVar("RequestType", bound=BaseModel)
ResponseType = TypeVar("ResponseType", bound=BaseModel)


class LLMClient(ABC):
    def __init__(self, config: LLMConfig, log_level):
        self.config = config
        self.logger = logger.setup(self.__class__.__name__, log_level)
        self.load()

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def get_response(self, request: RequestType) -> ResponseType:
        pass

    @abstractmethod
    def handle_error(self, error: Exception, message: str) -> None:
        pass

    @staticmethod
    def trim_message(message: str, max_tokens: int) -> str:
        return " ".join(message.split()[:max_tokens])
