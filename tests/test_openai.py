import pytest
from pydantic import ValidationError

from clients import OpenAIMockClient
from models.openai import *
from utils.config import LLMConfig

LOG_LEVEL = 10


@pytest.fixture
def client():
    config = LLMConfig(api_key="sk-key", model="gpt-4")
    return OpenAIMockClient(config, LOG_LEVEL)


# Config Tests
@pytest.mark.parametrize(
    "config_kwargs,expected_exception",
    [
        ({"model": "gpt-4"}, ValueError),  # Missing API key
        ({"api_key": "sk-key"}, ValueError),  # Missing model
    ]
)
def test_config_validation(config_kwargs, expected_exception):
    with pytest.raises(expected_exception):
        OpenAIMockClient(LLMConfig(**config_kwargs), LOG_LEVEL)


# Request Validation Tests
@pytest.mark.parametrize(
    "model_input,expected_error",
    [
        ("geppetto-4", r"Input should be .* \[type=literal_error"),  # Invalid model
        ("", r"Input should be .* \[type=literal_error"),  # Empty model
        (123, r"Input should be .* \[type=literal_error"),  # Invalid model type
    ]
)
def test_request_model_validation(model_input, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIRequest(
            messages=[OpenAIMessage(role="user", content="Hello!")],
            model=model_input,  # noqa
            max_tokens=42
        )


@pytest.mark.parametrize(
    "message,expected_error",
    [
        ({"role": "human", "content": "Hello!"}, r"Input should be 'user', 'assistant' or 'system'"),  # Invalid role
        ({"role": "user"}, r"Field required \[type=missing"),  # Missing content
        ({"content": "Role!"}, r"Field required \[type=missing"),  # Missing role
    ]
)
def test_message_structure_validation(message, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIRequest(
            messages=[OpenAIMessage(**message)],
            model="gpt-4",
            max_tokens=42
        )


@pytest.mark.parametrize(
    "messages,expected_error",
    [
        ([], r"List should have at least 1 item"),  # Empty messages list
        ({}, r"Input should be a valid list \[type=list_type"),  # Invalid messages type
    ]
)
def test_request_messages_validation(messages, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIRequest(
            messages=messages,  # noqa
            model="gpt-4",
            max_tokens=42
        )


@pytest.mark.parametrize(
    "max_tokens,expected_error",
    [
        (-1, r"Input should be greater than or equal to 1"),  # Invalid max_tokens
        (2049, r"Input should be less than or equal to 2048"),  # Invalid max_tokens
    ]
)
def test_request_max_tokens_validation(max_tokens, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIRequest(
            messages=[OpenAIMessage(role="user", content="Hello!")],
            model="gpt-4",
            max_tokens=max_tokens
        )


# Response Validation Tests
@pytest.mark.parametrize(
    "choice_kwargs,expected_error",
    [
        (  # Invalid finish reason
                {
                    "index": 0,
                    "finish_reason": "imlazy",
                    "message": OpenAIResponseMessage(role="assistant", content="Sorry!")
                },
                r"Input should be .* \[type=literal_error"
        ),
    ]
)
def test_response_choice_validation(choice_kwargs, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIChoice(**choice_kwargs)


@pytest.mark.parametrize(
    "message_kwargs,expected_error",
    [
        ({"role": "robot", "content": "Hello!"}, r"Input should be 'user', 'assistant' or 'system'"),  # Invalid role
        ({"role": "user"}, r"Field required \[type=missing"),  # Missing content
    ]
)
def test_response_message_validation(message_kwargs, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIResponseMessage(**message_kwargs)


@pytest.mark.parametrize(
    "usage_kwargs,expected_error",
    [
        (  # Invalid token counts
                {"prompt_tokens": "invalid", "completion_tokens": 10, "total_tokens": 20},
                r"Input should be a valid integer"
        ),
        (  # Negative tokens
                {"prompt_tokens": -1, "completion_tokens": 10, "total_tokens": 9},
                r"Input should be greater than or equal to 0"
        ),
    ]
)
def test_usage_validation(usage_kwargs, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIUsage(**usage_kwargs)


@pytest.mark.parametrize(
    "response_kwargs,expected_error",
    [
        (  # Invalid object type
                {
                    "id": "chatcmpl-12345",
                    "choices": [],
                    "created": 1678491234,
                    "model": "gpt-4",
                    "object": "invalid-object",
                    "usage": None
                },
                r"Input should be 'chat.completion'"
        ),
        (  # Empty choices list
                {
                    "id": "chatcmpl-12345",
                    "choices": [],
                    "created": 1678491234,
                    "model": "gpt-4",
                    "object": "chat.completion",
                    "usage": None
                },
                r"List should have at least 1 item"
        ),
    ]
)
def test_response_structure_validation(response_kwargs, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        OpenAIResponse(**response_kwargs)


# Client Tests
@pytest.mark.parametrize(
    "request_kwargs,expected_exception",
    [
        ({"messages": [], "model": "gpt-4"}, ValueError),  # Empty messages list
        ({"messages": [{"consent": "Denied!"}], "model": "gpt-4"}, ValidationError),  # Invalid message structure
        ({"messages": [OpenAIMessage(role="user", content="Hello!")], "model": "geppetto-4"}, ValueError),
        # Invalid model
    ]
)
def test_client_request_validation(client, request_kwargs, expected_exception):
    with pytest.raises(expected_exception):
        client.get_response(OpenAIRequest(**request_kwargs))


def test_valid_request(client):
    valid_request = OpenAIRequest(
        messages=[OpenAIMessage(role="user", content="Hello!")],
        model="gpt-4",
        max_tokens=42
    )
    valid_response = client.get_response(valid_request)

    assert valid_response.id is not None
