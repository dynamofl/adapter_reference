import os
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx
from openai import OpenAI
import uvicorn
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the OpenAI API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize FastAPI
app = FastAPI(
    title="OpenAI API Proxy",
    description="A proxy service for the OpenAI API",
    version="1.0.0",
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Models for request/response schemas
class ChatCompletionMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    user: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "OpenAI API Proxy is running. Use /v1 endpoints to access the API."}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Handle streaming responses
        if request.stream:
            return StreamingResponse(
                stream_chat_completions(request),
                media_type="text/event-stream"
            )
        
        # Process regular response
        response = client.chat.completions.create(
            model=request.model,
            messages=[{"role": msg.role, "content": msg.content, "name": msg.name} 
                     for msg in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def stream_chat_completions(request: ChatCompletionRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[{"role": msg.role, "content": msg.content, "name": msg.name} 
                     for msg in request.messages],
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=True,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        for chunk in response:
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in streaming chat completions: {str(e)}")
        yield f"data: {{'error': '{str(e)}'}}\n\n"

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingsRequest):
    try:
        response = client.embeddings.create(
            model=request.model,
            input=request.input,
            user=request.user
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    try:
        # Handle streaming responses
        if request.stream:
            return StreamingResponse(
                stream_completions(request),
                media_type="text/event-stream"
            )
        
        # Process regular response
        response = client.completions.create(
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=request.stream,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

async def stream_completions(request: CompletionRequest):
    try:
        response = client.completions.create(
            model=request.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stream=True,
            logprobs=request.logprobs,
            echo=request.echo,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user
        )
        
        for chunk in response:
            yield f"data: {chunk.model_dump_json()}\n\n"
        
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Error in streaming completions: {str(e)}")
        yield f"data: {{'error': '{str(e)}'}}\n\n"

# You can add more OpenAI API endpoints as needed

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
