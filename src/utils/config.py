from pydantic import field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    api_key: str
    model: str

    @field_validator('api_key')
    def api_key_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('API key must not be empty')
        return v

    @field_validator('model')
    def model_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Model name must not be empty')
        return v
