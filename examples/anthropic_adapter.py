"""
Anthropic Claude API Adapter Example

This demonstrates how to adapt the OpenAI API proxy to work with Anthropic's API.
To use this example:
1. Install the Anthropic Python SDK: pip install anthropic
2. Set ANTHROPIC_API_KEY in your .env file
3. Run this adapter with: python examples/anthropic_adapter.py
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import asyncio

import anthropic
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Import from the main app for reuse
from app import (
    RequestLoggingMiddleware, 
    get_api_key, 
    ChatCompletionMessage, 
    ChatCompletionRequest,
    get_error_details_and_status,
)
from config import config, AppConfig

# Configure Anthropic-specific settings
class AnthropicConfig(AppConfig):
    anthropic_api_key: str
    anthropic_model_map: Dict[str, str] = Field(default_factory=lambda: {
        "gpt-3.5-turbo": "claude-3-sonnet-20240229",
        "gpt-4": "claude-3-opus-20240229",
        "gpt-4-turbo": "claude-3-opus-20240229",
    })
    
    @property
    def anthropic_client(self):
        return anthropic.Anthropic(api_key=self.anthropic_api_key)

# Load Anthropic-specific config
def load_anthropic_config():
    # Get existing config and extend it
    base_config = config
    
    # Required variables
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    # Create Anthropic config
    return AnthropicConfig(
        **base_config.model_dump(),
        anthropic_api_key=anthropic_api_key
    )

# Initialize app
anthropic_config = load_anthropic_config()
app = FastAPI(
    title="Anthropic API Adapter",
    description="OpenAI API-compatible adapter for Anthropic Claude",
    version="1.0.0",
)
app.add_middleware(RequestLoggingMiddleware)

# Map from OpenAI API format to Anthropic format
def map_to_anthropic_messages(openai_messages: List[ChatCompletionMessage]):
    """Convert OpenAI messages format to Anthropic messages format"""
    anthropic_messages = []
    
    for msg in openai_messages:
        role = msg.role
        if role == "system":
            # System messages are handled differently in Anthropic
            continue
        elif role == "assistant":
            anthropic_role = "assistant"
        else:
            anthropic_role = "user"
            
        anthropic_messages.append({
            "role": anthropic_role,
            "content": msg.content
        })
    
    # Extract system message if present
    system_messages = [msg for msg in openai_messages if msg.role == "system"]
    system_content = system_messages[0].content if system_messages else None
    
    return anthropic_messages, system_content

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key),
    x_request_id: Optional[str] = Header(None)
):
    """Create a chat completion with the Anthropic API"""
    try:
        # Map OpenAI model to Anthropic model
        anthropic_model = anthropic_config.anthropic_model_map.get(
            request.model, 
            "claude-3-sonnet-20240229"  # Default model
        )
        
        # Convert messages format
        anthropic_messages, system_content = map_to_anthropic_messages(request.messages)
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completions(request, anthropic_model, anthropic_messages, system_content),
                media_type="text/event-stream"
            )
        
        # Regular response
        client = anthropic_config.anthropic_client
        response = client.messages.create(
            model=anthropic_model,
            messages=anthropic_messages,
            system=system_content,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
        )
        
        # Convert to OpenAI format
        openai_response = {
            "id": f"chatcmpl-{response.id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content[0].text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        }
        
        return openai_response
    except Exception as e:
        error_details, status_code = get_error_details_and_status(e)
        raise HTTPException(
            status_code=status_code,
            detail=error_details
        )

async def stream_chat_completions(
    request: ChatCompletionRequest, 
    anthropic_model: str,
    anthropic_messages: List[Dict[str, str]],
    system_content: Optional[str]
):
    """Stream chat completions from the Anthropic API"""
    try:
        client = anthropic_config.anthropic_client
        
        with client.messages.stream(
            model=anthropic_model,
            messages=anthropic_messages,
            system=system_content,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
        ) as stream:
            for text in stream.text_stream:
                # Convert to OpenAI format
                chunk = {
                    "id": f"chatcmpl-{stream.id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": text,
                            },
                            "finish_reason": None,
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
            # Send final chunk
            final_chunk = {
                "id": f"chatcmpl-{stream.id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            
    except Exception as e:
        error_details, _ = get_error_details_and_status(e)
        error_json = json.dumps(error_details)
        yield f"data: {error_json}\n\n"

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "examples.anthropic_adapter:app", 
        host="0.0.0.0", 
        port=anthropic_config.port,
        reload=True
    )