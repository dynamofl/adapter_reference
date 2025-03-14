"""
Mistral API Adapter Example

This demonstrates how to adapt the OpenAI API proxy to work with Mistral's API.
To use this example:
1. Install the Mistral Python SDK: pip install mistralai
2. Set MISTRAL_API_KEY in your .env file
3. Run this adapter with: python examples/mistral_adapter.py
"""

import os
import json
import time
from typing import Dict, List, Optional, Any

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
from fastapi.responses import StreamingResponse, JSONResponse

# Import from the main app for reuse
from app import (
    RequestLoggingMiddleware, 
    get_api_key, 
    ChatCompletionMessage, 
    ChatCompletionRequest,
    get_error_details_and_status,
)
from config import config, AppConfig

# Configure Mistral-specific settings
class MistralConfig(AppConfig):
    mistral_api_key: str
    mistral_model_map: Dict[str, str] = {
        "gpt-3.5-turbo": "mistral-small-latest",
        "gpt-4": "mistral-medium-latest",
        "gpt-4-turbo": "mistral-large-latest",
    }
    
    @property
    def mistral_client(self):
        return MistralClient(api_key=self.mistral_api_key)

# Load Mistral-specific config
def load_mistral_config():
    # Get existing config and extend it
    base_config = config
    
    # Required variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is required")
    
    # Create Mistral config
    return MistralConfig(
        **base_config.model_dump(),
        mistral_api_key=mistral_api_key
    )

# Initialize app
mistral_config = load_mistral_config()
app = FastAPI(
    title="Mistral API Adapter",
    description="OpenAI API-compatible adapter for Mistral AI",
    version="1.0.0",
)
app.add_middleware(RequestLoggingMiddleware)

# Map from OpenAI API format to Mistral format
def map_to_mistral_messages(openai_messages: List[ChatCompletionMessage]):
    """Convert OpenAI messages format to Mistral messages format"""
    mistral_messages = []
    
    for msg in openai_messages:
        role = msg.role
        # Mistral doesn't support function or tool roles, so skip them
        if role in ["function", "tool"]:
            continue
            
        mistral_messages.append(
            ChatMessage(role=role, content=msg.content)
        )
    
    return mistral_messages

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key),
    x_request_id: Optional[str] = Header(None)
):
    """Create a chat completion with the Mistral API"""
    try:
        # Map OpenAI model to Mistral model
        mistral_model = mistral_config.mistral_model_map.get(
            request.model, 
            "mistral-small-latest"  # Default model
        )
        
        # Convert messages format
        mistral_messages = map_to_mistral_messages(request.messages)
        
        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_chat_completions(request, mistral_model, mistral_messages),
                media_type="text/event-stream"
            )
        
        # Regular response
        client = mistral_config.mistral_client
        response = client.chat(
            model=mistral_model,
            messages=mistral_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
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
                        "content": response.choices[0].message.content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
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
    mistral_model: str,
    mistral_messages: List[ChatMessage],
):
    """Stream chat completions from the Mistral API"""
    try:
        client = mistral_config.mistral_client
        
        stream = client.chat_stream(
            model=mistral_model,
            messages=mistral_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        for chunk in stream:
            # Convert to OpenAI format
            openai_chunk = {
                "id": f"chatcmpl-{chunk.id}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk.choices[0].delta.content,
                        },
                        "finish_reason": chunk.choices[0].finish_reason,
                    }
                ]
            }
            yield f"data: {json.dumps(openai_chunk)}\n\n"
            
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
        "examples.mistral_adapter:app", 
        host="0.0.0.0", 
        port=mistral_config.port,
        reload=True
    )