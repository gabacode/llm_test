from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class AnthropicContent(BaseModel):
    type: Literal['text', 'image'] = Field(
        ...,
        description="The type of content block, either 'text' or 'image'."
    )
    text: Optional[str] = Field(
        None,
        description="The text content."
    )


class AnthropicMessage(BaseModel):
    role: Literal['user', 'assistant'] = Field(
        ...,
        description="The role of the message sender, either 'user' or 'assistant'."
    )
    content: Union[str, List[AnthropicContent]] = Field(
        ...,
        description=(
            "The content of the message. Can be a string for text-only messages or a list of content blocks for multimodal messages."
        )
    )


class AnthropicRequest(BaseModel):
    model: Literal["claude-3-5-sonnet-20241022"] = Field(
        ...,
        description="ID of the model to use."
    )
    system: Optional[str] = Field(
        None,
        description="A system prompt providing context and instructions to Claude."
    )
    messages: List[AnthropicMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far.",
        min_length=1
    )
    temperature: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="The degree of randomness in the model's output. Range: 0.0 to 1.0."
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "model": "claude-3-5-sonnet-20241022",
                "system": "You are not Claude, but you look like it.",
                "messages": [
                    {"role": "user", "content": "Hello, Claude"},
                    {"role": "assistant", "content": "Hello! How can I assist you today?"},
                    {"role": "user", "content": "I was looking for your swagger but could not find it!"},
                ],
                "temperature": 0.7,
            }
        }
