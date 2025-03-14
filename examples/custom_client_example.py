"""
Custom OpenAI Client Implementation Example

This example demonstrates how to create a custom client implementation
that maintains the same interface as the OpenAI client but has different
internal authentication and routing logic.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types import Completion

# Import configuration
from config import config

# Configure logging
logger = logging.getLogger(__name__)

class CustomOpenAIClient:
    """
    Custom OpenAI client that implements the same interface but with different internal logic.
    
    This example client implements the same methods as the OpenAI client,
    but adds custom authentication, logging, and potentially routing.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", 
                 enterprise_id: Optional[str] = None, custom_headers: Optional[Dict[str, str]] = None):
        """
        Initialize the custom client with your organization's specific requirements.
        
        Args:
            api_key: The OpenAI API key
            base_url: The base URL for the OpenAI API
            enterprise_id: Your organization's enterprise ID (example custom parameter)
            custom_headers: Additional headers to include in requests
        """
        # You would import your actual OpenAI client here
        from openai import OpenAI
        
        # Store the original client
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Store custom parameters
        self.enterprise_id = enterprise_id
        self.custom_headers = custom_headers or {}
        
        logger.info(f"Initialized custom OpenAI client with enterprise_id: {enterprise_id}")
    
    @property
    def chat(self):
        """
        Access the chat completions API with custom logic.
        """
        return CustomChatCompletions(self)
    
    @property
    def completions(self):
        """
        Access the completions API with custom logic.
        """
        return CustomCompletions(self)
    
    @property
    def models(self):
        """
        Pass through to the original models API.
        """
        return self._client.models


class CustomChatCompletions:
    """
    Custom implementation of the chat completions API.
    """
    
    def __init__(self, client):
        """
        Initialize with a reference to the parent client.
        """
        self._client = client
        self._original = client._client.chat.completions
    
    def create(self, model: str, messages: List[Dict[str, str]], **kwargs):
        """
        Create a chat completion with custom logic.
        
        This method demonstrates how you could add custom logic before
        passing the request to the actual OpenAI API.
        """
        # Log request for analytics
        logger.info(f"Chat completion request: model={model}, messages={len(messages)} messages")
        
        # Example: Add enterprise ID to user field if not specified
        if 'user' not in kwargs and self._client.enterprise_id:
            kwargs['user'] = f"enterprise-{self._client.enterprise_id}"
        
        # Example: Add custom routing logic based on model
        if model.startswith("enterprise-"):
            # Route to a custom endpoint for enterprise models
            custom_model = model.replace("enterprise-", "")
            
            # Generate a mocked response for the example
            # In a real implementation, you would call your custom endpoint
            return self._mock_response(custom_model, messages)
        
        # For standard models, pass through to the original API
        return self._original.create(model=model, messages=messages, **kwargs)
    
    def _mock_response(self, model: str, messages: List[Dict[str, str]]) -> ChatCompletion:
        """
        Mock a chat completion response for demonstration purposes.
        
        In a real implementation, this would call your custom API.
        """
        return ChatCompletion(
            id="custom-chat-" + os.urandom(8).hex(),
            object="chat.completion",
            created=1234567890,
            model=model,
            choices=[{
                "index": 0,
                "message": ChatCompletionMessage(
                    role="assistant",
                    content=f"This is a response from custom enterprise model: {model}"
                ),
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        )


class CustomCompletions:
    """
    Custom implementation of the completions API.
    """
    
    def __init__(self, client):
        """
        Initialize with a reference to the parent client.
        """
        self._client = client
        self._original = client._client.completions
    
    def create(self, model: str, prompt: str, **kwargs):
        """
        Create a completion with custom logic.
        """
        # Log request for analytics
        logger.info(f"Completion request: model={model}, prompt={prompt[:50]}...")
        
        # Example: Add enterprise ID to user field if not specified
        if 'user' not in kwargs and self._client.enterprise_id:
            kwargs['user'] = f"enterprise-{self._client.enterprise_id}"
        
        # Example: Add custom routing logic based on model
        if model.startswith("enterprise-"):
            # Route to a custom endpoint for enterprise models
            custom_model = model.replace("enterprise-", "")
            
            # Generate a mocked response for the example
            # In a real implementation, you would call your custom endpoint
            return self._mock_response(custom_model, prompt)
        
        # For standard models, pass through to the original API
        return self._original.create(model=model, prompt=prompt, **kwargs)
    
    def _mock_response(self, model: str, prompt: str) -> Completion:
        """
        Mock a completion response for demonstration purposes.
        
        In a real implementation, this would call your custom API.
        """
        return Completion(
            id="custom-completion-" + os.urandom(8).hex(),
            object="text_completion",
            created=1234567890,
            model=model,
            choices=[{
                "text": f"This is a response from custom enterprise model: {model}",
                "index": 0,
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        )


# Example of how to use the custom client in the main application
def get_custom_openai_client():
    """
    Create a custom OpenAI client to use in your application.
    
    This would replace the get_openai_client() function in app.py.
    """
    return CustomOpenAIClient(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        enterprise_id=os.getenv("ENTERPRISE_ID", "default"),
        custom_headers={"X-Enterprise-Auth": os.getenv("ENTERPRISE_AUTH_TOKEN", "")}
    )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client
    client = get_custom_openai_client()
    
    # Example chat completion with standard model
    chat_response = client.chat.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print("Standard model response:", chat_response.choices[0].message.content)
    
    # Example chat completion with enterprise model
    enterprise_response = client.chat.create(
        model="enterprise-custom-model",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print("Enterprise model response:", enterprise_response.choices[0].message.content)