import logging

import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from clients import OpenAIMockClient, AnthropicMockClient
from models.anthropic import AnthropicResponse, AnthropicRequest
from models.openai import OpenAIResponse, OpenAIRequest
from utils.config import LLMConfig

app = FastAPI()


@app.get("/", summary="Root", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/chat/completions", summary="Create Chat Completion", tags=["OpenAI"],
          response_model=OpenAIResponse)
def create_chat_completion(request: OpenAIRequest):
    config = LLMConfig(api_key="sk-key", model=request.model)
    client = OpenAIMockClient(config=config, log_level=logging.DEBUG)
    response = client.get_response(request=request)
    return response.model_dump()


@app.post("/claude/completions", summary="Create Claude Completion", tags=["Anthropic"],
          response_model=AnthropicResponse)
def create_claude_completion(request: AnthropicRequest):
    config = LLMConfig(api_key="cl-key", model=request.model)
    client = AnthropicMockClient(config=config, log_level=logging.DEBUG)
    response = client.get_response(request=request)
    return response.model_dump()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
