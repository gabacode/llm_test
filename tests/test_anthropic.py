import pytest
from pydantic import ValidationError

from clients import AnthropicMockClient
from models.anthropic import AnthropicRequest, AnthropicMessage
from utils.config import LLMConfig

LOG_LEVEL = 10


@pytest.fixture
def client():
    config = LLMConfig(api_key="sk-key", model="claude-3-5-sonnet-20241022")
    return AnthropicMockClient(config, LOG_LEVEL)


# Config Tests
@pytest.mark.parametrize(
    "config_kwargs,expected_exception",
    [
        ({"model": "claude-3-5-sonnet-20241022"}, ValueError),  # Missing API key
        ({"api_key": "cl-key"}, ValueError),  # Missing model
    ]
)
def test_config_validation(config_kwargs, expected_exception):
    with pytest.raises(expected_exception):
        AnthropicMockClient(LLMConfig(**config_kwargs), LOG_LEVEL)


# Request Validation Tests
@pytest.mark.parametrize(
    "model_input,expected_error",
    [
        ("claudio", r"Input should be .* \[type=literal_error"),  # Invalid model
        ("", r"Input should be .* \[type=literal_error"),  # Empty model
        (123, r"Input should be .* \[type=literal_error"),  # Invalid model type
    ]
)
def test_request_model_validation(model_input, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        AnthropicRequest(
            messages=[AnthropicMessage(role="user", content="Hello!")],
            model=model_input,  # noqa
            system="",
            temperature=0,
            max_tokens=42
        )


@pytest.mark.parametrize(
    "message,expected_error",
    [
        ("invalid-type", r"Input should be a valid list \[type=list_type"),  # Invalid message type
        ([], r"List should have at least 1 item"),  # Empty messages list
    ]
)
def test_request_messages_validation(message, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        AnthropicRequest(
            messages=message,
            model="claude-3-5-sonnet-20241022",
            system="",
            temperature=0,
            max_tokens=42
        )


@pytest.mark.parametrize(
    "message_structure,expected_error",
    [
        ({"role": "human", "content": "Hello!"}, r"Input should be 'user' or 'assistant'"),  # Invalid role
        ({"role": "user"}, r"Field required \[type=missing"),  # Missing content
    ]
)
def test_message_structure_validation(message_structure, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        AnthropicRequest(
            messages=[AnthropicMessage(**message_structure)],
            model="claude-3-5-sonnet-20241022",
            system="",
            temperature=0,
            max_tokens=42
        )


# Further Tests for Edge Cases
@pytest.mark.parametrize(
    "temperature_value,expected_error",
    [
        (-1, r"Input should be greater than or equal to 0"),  # Negative temperature
        (2.5, r"Input should be less than or equal to 1"),  # Exceeding valid range
    ]
)
def test_temperature_validation(temperature_value, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        AnthropicRequest(
            messages=[AnthropicMessage(role="user", content="Hello!")],
            model="claude-3-5-sonnet-20241022",
            system="",
            temperature=temperature_value,
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
        AnthropicRequest(
            messages=[AnthropicMessage(role="user", content="Hello!")],
            model="claude-3-5-sonnet-20241022",
            system="",
            temperature=0,
            max_tokens=max_tokens
        )


# Client Tests
@pytest.mark.parametrize(
    "request_kwargs,expected_exception",
    [
        ({"messages": [], "model": "claude-3-5-sonnet-20241022", "system": "", "temperature": 0}, ValueError),
        # Empty Messages
        ({"messages": [AnthropicMessage(role="user", content="Hi!")], "model": "", "system": "", "temperature": 0},
         # Empty Model
         ValidationError),
    ]
)
def test_client_request_validation(client, request_kwargs, expected_exception):
    with pytest.raises(expected_exception):
        client.get_response(AnthropicRequest(**request_kwargs))


def test_invalid_role_in_message(client):
    invalid_message = {"role": "human", "content": "Hi!"}

    invalid_request = {
        "messages": [invalid_message],
        "model": "claude-3-5-sonnet-20241022",
        "system": "",
        "temperature": 0
    }
    with pytest.raises(ValidationError):
        AnthropicRequest(**invalid_request)


def test_valid_request(client):
    valid_request = AnthropicRequest(
        messages=[AnthropicMessage(role="user", content="Hello!")],
        model="claude-3-5-sonnet-20241022",
        system="You are Claudio",
        temperature=0,
        max_tokens=42
    )
    response = client.get_response(valid_request)

    assert response.id is not None
