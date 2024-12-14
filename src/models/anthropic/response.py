from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AnthropicContent(BaseModel):
    type: Literal['text', 'image'] = Field(
        ...,
        description="The type of content block: 'text', or 'image'."
    )
    text: Optional[str] = Field(
        None,
        description="The text content."
    )


class AnthropicUsage(BaseModel):
    input_tokens: int = Field(
        ...,
        description="The number of tokens used in the input prompt."
    )
    output_tokens: int = Field(
        ...,
        description="The number of tokens used in the output completion."
    )


class AnthropicResponse(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the completion response."
    )
    type: Literal['message'] = Field(
        'message',
        description="The type of the response object, always 'message'."
    )
    role: Literal['assistant'] = Field(
        'assistant',
        description="The role of the message sender, always 'assistant'."
    )
    content: List[AnthropicContent] = Field(
        ...,
        description=(
            "A list of content blocks composing the assistant's response. Each block can "
            "represent a text response, an image, a tool usage, or a tool result."
        )
    )
    model: str = Field(
        ...,
        description="The model ID used for generating the completion."
    )
    stop_reason: Optional[str] = Field(
        None,
        description="The reason why the generation of the response was stopped."
    )
    usage: Optional[AnthropicUsage] = Field(
        None,
        description=(
            "Optional information on token usage for the completion."
        )
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "claude-response-12345",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello! How can I assist you today?"
                    },
                ],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 15,
                },
            }
        }
