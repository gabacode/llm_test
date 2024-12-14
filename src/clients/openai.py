from time import time
from typing import List
from uuid import uuid1

from pydantic import ValidationError

from clients.llm_client import LLMClient
from models.openai import OpenAIRequest, OpenAIResponse, OpenAIMessage, OpenAIUsage


class OpenAIMockClient(LLMClient):
    def __init__(self, config, log_level):
        super().__init__(config, log_level)

    def load(self):
        self.logger.debug("Loaded 🚀")

    @staticmethod
    def calculate_usage(messages: List[OpenAIMessage], response: str) -> OpenAIUsage:
        prompt_tokens = sum(
            len(message.role) + len(message.content.split()) for message in messages
        )
        completion_tokens = len(response.split())
        total = prompt_tokens + completion_tokens
        return OpenAIUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total)

    def get_response(self, request: OpenAIRequest) -> OpenAIResponse:
        try:
            self.logger.debug(f"Request: {request.model_dump_json()}")
            answer = "I didn't understand that. Can you please join our premium program?"
            answer = self.trim_message(answer, request.max_tokens)
            usage = self.calculate_usage(request.messages, answer)

            response = OpenAIResponse(
                id=uuid1().hex,
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"{answer}"},
                    }
                ],
                created=int(time()),
                model=request.model,
                object="chat.completion",
                usage=usage
            )
            self.logger.debug(f"Response: {response.model_dump_json()}")

            return response
        except ValidationError as e:
            self.handle_error(e, "Validation error.")
            raise
        except Exception as e:
            self.handle_error(e, "An unexpected error occurred.")
            raise

    def handle_error(self, error: Exception, message: str):
        self.logger.error(f"Error: {message} | Exception: {error}")
        raise error
