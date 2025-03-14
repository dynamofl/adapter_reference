"""
OpenAI API Proxy Service

This service provides a focused adapter for OpenAI's chat and completion APIs.
It handles authentication, error handling, request validation, and metrics.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
import httpx
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import uvicorn
import logging
import secrets
from datetime import datetime

# Import configuration from config.py
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OpenAI API Proxy",
    description="A focused text generation proxy service for OpenAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize OpenAI client
def get_openai_client():
    """
    Create an OpenAI client using the configured API key and base URL.
    
    This creates a standard OpenAI client that connects to the actual OpenAI API.
    When customers implement their own client with the same interface, they can
    replace this with their own implementation while keeping the same API endpoints.
    """
    return OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url
    )

# Middleware for request logging
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        start_time = time.time()
        
        request_id = secrets.token_hex(8)
        
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
        
        return response

# Add middleware
app.add_middleware(RequestLoggingMiddleware)

# Authentication dependency
async def get_api_key(
    api_key: str = Depends(api_key_header)
) -> str:
    """
    Validate the API key provided in the request.
    
    This function is used as a dependency for protected endpoints.
    If authentication is disabled, it allows all requests through.
    
    Args:
        api_key: The API key from the X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If authentication is enabled and the API key is invalid
    """
    if not config.enable_auth:
        return None
    
    if api_key is None or api_key != config.api_key_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key

# Initialize OpenAI client as a dependency
client = get_openai_client()

# Response models for error handling
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None

# Models for request/response schemas
class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ['system', 'user', 'assistant', 'function', 'tool']
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of {allowed_roles}")
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one message is required")
        return v

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(16, ge=1, le=8192)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    stream: Optional[bool] = False
    logprobs: Optional[int] = Field(None, ge=0, le=5)
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Prompt list cannot be empty")
        return v

# Utils for handling errors
def get_error_details_and_status(error: Exception) -> Tuple[Dict[str, Any], int]:
    """
    Process an exception and return appropriate error details and HTTP status code.
    
    Args:
        error: The exception that occurred
        
    Returns:
        tuple: (error_details, http_status_code)
    """
    if isinstance(error, APIError):
        # Check if there's a status code in the error
        status_code = getattr(error, "status_code", 500)
        return {
            "error": error.__class__.__name__,
            "detail": "OpenAI API error occurred",
            "code": getattr(error, "code", "unknown_error")
        }, status_code
    elif isinstance(error, RateLimitError):
        return {
            "error": "RateLimitError",
            "detail": "OpenAI API rate limit exceeded",
            "code": "rate_limit_exceeded"
        }, status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(error, AuthenticationError):
        return {
            "error": "AuthenticationError",
            "detail": "Authentication with OpenAI API failed",
            "code": "authentication_failed"
        }, status.HTTP_401_UNAUTHORIZED
    elif isinstance(error, ValidationError):
        return {
            "error": "ValidationError",
            "detail": str(error),
            "code": "validation_error"
        }, status.HTTP_422_UNPROCESSABLE_ENTITY
    else:
        # Generic error without details
        return {
            "error": "ServerError",
            "detail": "An unexpected error occurred",
            "code": "internal_server_error"
        }, status.HTTP_500_INTERNAL_SERVER_ERROR

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OpenAI API Proxy is running. Use /v1 endpoints to access the API.",
        "documentation": "/docs",
        "health": "/health"
    }

# Chat completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(get_api_key),
    x_request_id: Optional[str] = Header(None)
):
    """
    Create a chat completion with the OpenAI API.
    
    This endpoint proxies requests to OpenAI's chat completions API,
    with additional validation, error handling, and model restrictions.
    """
    try:
        # Check if the requested model is allowed
        if config.allowed_models and request.model not in config.allowed_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ModelNotAllowedError",
                    "detail": f"Model '{request.model}' is not allowed. Allowed models: {', '.join(config.allowed_models)}",
                    "code": "model_not_allowed"
                }
            )
        
        # Handle streaming responses
        if request.stream:
            return StreamingResponse(
                stream_chat_completions(request),
                media_type="text/event-stream"
            )
        
        # Process regular response
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in request.messages
        ]

        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
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
        error_details, status_code = get_error_details_and_status(e)
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(
            status_code=status_code,
            detail=error_details
        )

async def stream_chat_completions(request: ChatCompletionRequest):
    """
    Stream chat completions from the OpenAI API.
    
    This function handles streaming responses, allowing for real-time
    updates as the completion is generated.
    """
    try:
        # Check if the requested model is allowed
        if config.allowed_models and request.model not in config.allowed_models:
            error_details = {
                "error": "ModelNotAllowedError",
                "detail": f"Model '{request.model}' is not allowed. Allowed models: {', '.join(config.allowed_models)}",
                "code": "model_not_allowed"
            }
            error_json = json.dumps(error_details)
            yield f"data: {error_json}\n\n"
            return
            
        messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in request.messages
        ]

        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
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
        error_details, _ = get_error_details_and_status(e)
        logger.error(f"Error in streaming chat completions: {str(e)}")
        error_json = json.dumps(error_details)
        yield f"data: {error_json}\n\n"

# Completions endpoint
@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    api_key: str = Depends(get_api_key),
    x_request_id: Optional[str] = Header(None)
):
    """
    Create completions with the OpenAI API.
    
    This endpoint supports both the legacy completions API
    and provides the same validation and error handling as the chat endpoint.
    """
    try:
        # Check if the requested model is allowed
        if config.allowed_models and request.model not in config.allowed_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ModelNotAllowedError",
                    "detail": f"Model '{request.model}' is not allowed. Allowed models: {', '.join(config.allowed_models)}",
                    "code": "model_not_allowed"
                }
            )
            
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
        error_details, status_code = get_error_details_and_status(e)
        logger.error(f"Error in completions: {str(e)}")
        raise HTTPException(
            status_code=status_code,
            detail=error_details
        )

async def stream_completions(request: CompletionRequest):
    """
    Stream completions from the OpenAI API.
    
    This function handles streaming for the legacy completions endpoint,
    with the same validation and error handling as the chat endpoint.
    """
    try:
        # Check if the requested model is allowed
        if config.allowed_models and request.model not in config.allowed_models:
            error_details = {
                "error": "ModelNotAllowedError",
                "detail": f"Model '{request.model}' is not allowed. Allowed models: {', '.join(config.allowed_models)}",
                "code": "model_not_allowed"
            }
            error_json = json.dumps(error_details)
            yield f"data: {error_json}\n\n"
            return
            
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
        error_details, _ = get_error_details_and_status(e)
        logger.error(f"Error in streaming completions: {str(e)}")
        error_json = json.dumps(error_details)
        yield f"data: {error_json}\n\n"

# Metrics endpoint for monitoring
@app.get("/metrics", dependencies=[Depends(get_api_key)])
async def metrics():
    """Provide metrics about API usage"""
    return {
        "uptime_seconds": time.time() - startup_time,
        "version": "1.0.0"
    }

# Initialize startup time for uptime tracking
startup_time = time.time()

# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors and return friendly responses"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "code": "validation_error"
        }
    )

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI request validation errors and return in consistent format"""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        error_messages.append(f"{error.get('msg')} at {'.'.join(str(loc) for loc in error.get('loc', []))}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": "; ".join(error_messages),
            "code": "validation_error"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return standardized responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "code": f"http_{exc.status_code}"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions with appropriate status codes"""
    logger.exception("Unhandled exception occurred", exc_info=exc)
    error_details, status_code = get_error_details_and_status(exc)
    return JSONResponse(
        status_code=status_code,
        content=error_details
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    logger.info("Starting OpenAI API Proxy")
    logger.info(f"Authentication enabled: {config.enable_auth}")
    
    # Log configuration information
    if config.allowed_models:
        logger.info(f"Allowed models: {', '.join(config.allowed_models)}")
    else:
        logger.info("All models are allowed")
        
    logger.info(f"Using OpenAI base URL: {config.openai_base_url}")
    
    # Verify OpenAI API key on startup
    try:
        test_client = get_openai_client()
        models = test_client.models.list()
        logger.info(f"Successfully connected to OpenAI API. {len(models.data)} models available.")
        
        # Log available models for reference
        model_names = [model.id for model in models.data]
        logger.info(f"Available models: {', '.join(model_names[:10])}{'...' if len(model_names) > 10 else ''}")
    except Exception as e:
        logger.error(f"Failed to connect to OpenAI API: {str(e)}")
        # Continue startup even if OpenAI connection fails

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    logger.info("Shutting down OpenAI API Proxy")

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=config.port, 
        reload=True,
        log_level="info"
    )
