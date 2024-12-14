from typing import List, Literal

from pydantic import BaseModel, Field


class OpenAIMessage(BaseModel):
    role: Literal['user', 'assistant', 'system'] = Field(
        ...,
        description="The role of the message sender (e.g., 'user', 'assistant', or 'system')."
    )
    content: str = Field(
        ...,
        description="The content of the message."
    )


class OpenAIRequest(BaseModel):
    messages: List[OpenAIMessage] = Field(
        ...,
        description=(
            "A list of messages comprising the conversation so far. Depending on the "
            "model you use, different message types (modalities) are supported, like "
            "text, images, and audio."
        ),
        min_length=1
    )
    model: Literal[
        "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12",
        "gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
        "gpt-4o-realtime-preview", "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-10-01",
        "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-0125-preview",
        "gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-vision-preview",
        "gpt-4", "gpt-4-0314", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0314",
        "gpt-4-32k-0613", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613"
    ] = Field(
        ...,
        description=(
            "ID of the model to use. See the model endpoint compatibility table for "
            "details on which models work with the Chat API."
        )
    )

    class ConfigDict:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi! How can I help you today?"},
                    {"role": "user", "content": "How to give you all my money?"},
                ],
                "model": "gpt-4o"
            }
        }
