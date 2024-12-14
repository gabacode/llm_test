from typing import List
from uuid import uuid1

from pydantic import ValidationError

from clients.llm_client import LLMClient
from models.anthropic import AnthropicRequest, AnthropicResponse, AnthropicMessage, AnthropicContent, AnthropicUsage


class AnthropicMockClient(LLMClient):
    def __init__(self, config, log_level):
        super().__init__(config, log_level)

    def load(self):
        self.logger.debug("Loaded 🚀")

    @staticmethod
    def calculate_usage(messages: List[AnthropicMessage], response: List[str]) -> AnthropicUsage:
        prompt_tokens = sum(
            len(message.role) + (len(message.content) if isinstance(message.content, str) else sum(
                len(block.text or "") for block in message.content))
            for message in messages
        )
        completion_tokens = sum(len(block) for block in response)
        return AnthropicUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens)

    def get_response(self, request: AnthropicRequest) -> AnthropicResponse:
        try:
            self.logger.debug(f"Request: {request.model_dump_json()}")

            mock_content = [AnthropicContent(type="text", text="Hello!")]
            usage = self.calculate_usage(request.messages, [block.text for block in mock_content])

            response = AnthropicResponse(
                id=uuid1().hex,
                type="message",
                role="assistant",
                content=mock_content,
                usage=usage,
                stop_reason="end_turn",
                model=request.model
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
