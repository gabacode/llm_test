from typing import Literal, Optional

from pydantic import BaseModel, Field, conlist


class OpenAIResponseMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="The role of the message sender (e.g., 'user', 'assistant', or 'system')."
    )
    content: str = Field(
        ...,
        description="The content of the message."
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "role": "assistant",
                "content": "This is a response from the assistant."
            }
        }


class OpenAIChoice(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = Field(
        ...,
        description=(
            "The reason the model stopped generating tokens. Values can include: "
            "'stop' for natural stop points, 'length' if the maximum token limit was reached, "
            "'content_filter' if flagged, 'tool_calls' if the model invoked a tool, or 'function_call' "
            "(deprecated)."
        )
    )
    index: int = Field(
        ...,
        description="The index of the choice in the list of choices."
    )
    message: OpenAIResponseMessage = Field(
        ...,
        description="The message content of the choice."
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is an example response."
                },
            }
        }


class OpenAIUsage(BaseModel):
    prompt_tokens: int = Field(ge=0, description="Prompt token count must be >= 0")
    completion_tokens: int = Field(ge=0, description="Completion token count must be >= 0")
    total_tokens: int = Field(ge=0, description="Total token count must be >= 0")

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            }
        }


class OpenAIResponse(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the chat completion."
    )
    choices: conlist(OpenAIChoice, min_length=1) = Field(
        ...,
        description="A list of chat completion choices. Must contain at least one item."
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) when the chat completion was created."
    )
    model: str = Field(
        ...,
        description="The model used for the chat completion."
    )
    object: Literal["chat.completion"] = Field(
        ...,
        description="The object type, always 'chat.completion'."
    )
    usage: Optional[OpenAIUsage] = Field(
        None,
        description="Usage information for the completion, including token counts."
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "id": "chatcmpl-12345",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Here is a response."
                        }
                    }
                ],
                "created": 1678491234,
                "model": "gpt-4",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25
                }
            }
        }
