# OpenAI Adapter Reference

A reference implementation for building OpenAI API-compatible services. This adapter implements the OpenAI API interface and can be used as a starting point for creating custom implementations that maintain the same API but with different internal authentication, routing, or business logic.

## Features

- **Focused API**: Supports only the core completions endpoints
- **Authentication**: Optional API key authentication for securing your proxy
- **Request Validation**: Validates all requests before sending to the AI provider
- **Improved Error Handling**: Returns appropriate HTTP status codes and sanitized error messages
- **Model Restrictions**: Optionally restrict which models can be used
- **Custom Base URL**: Connect to different OpenAI-compatible endpoints
- **Metrics**: Built-in metrics endpoint for monitoring
- **Streaming Support**: Fully supports streaming responses for chat and completions
- **Health Check**: Built-in health check endpoint for monitoring
- **Extensibility**: Example adapters for other AI providers

## Supported Endpoints

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Text completions API (legacy)

Note: Other OpenAI endpoints like embeddings, image generation, and model listing are not supported in this focused adapter.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key (or API key for another provider if using an adapter)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/dynamofl/adapter_reference.git
cd adapter_reference
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Edit the `.env` file and add your API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running Locally

```bash
python app.py
```

The server will start at http://localhost:8000 by default.

### Using Docker

```bash
docker compose up --build
```

## Configuration

The following environment variables can be used to configure the adapter:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) | - |
| `OPENAI_BASE_URL` | Custom base URL for OpenAI API | `https://api.openai.com/v1` |
| `ENABLE_AUTH` | Enable API key authentication | `false` |
| `API_KEY_SECRET` | API key for authentication when enabled | Random generated |
| `PORT` | Port to run the server on | `8000` |
| `ALLOWED_MODELS` | Comma-separated list of allowed models | All models allowed |

## Using the API

### Making API Requests

You can make requests to the API without authentication:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

Authentication is disabled by default but can be enabled by uncommenting the authentication code in `app.py` and setting `ENABLE_AUTH=true` in your `.env` file.

### Using with the OpenAI Python client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    # No API key needed by default since authentication is disabled
    # If you enable authentication, include your API key:
    # api_key="your_api_key_here"
)

# Chat completions
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)

# Completions (legacy)
completion_response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Hello, how are you?",
    max_tokens=50
)
print(completion_response.choices[0].text)
```

## Creating Custom Client Implementations

This reference implementation is designed to help you build services that expose the same API interface as OpenAI but with your own customized client implementation. Here's how to use it:

### Understanding the Architecture

The reference implementation has three main components:

1. **API Endpoints**: FastAPI routes that expose OpenAI-compatible endpoints
2. **Client Implementation**: The OpenAI client that handles actual API calls
3. **Validation & Error Handling**: Logic that ensures requests and responses follow the OpenAI format

### Customizing for Your Own Implementation

To create your own client implementation that maintains OpenAI API compatibility:

1. **Fork this repo**: Use it as a starting point for your implementation
2. **Replace the client**: Modify the `get_openai_client()` function in `app.py` to return your custom client
3. **Customize authentication**: Update authentication as needed for your use case
4. **Add business logic**: Insert any additional logic like request modification, logging, etc.

Example custom client implementation:

```python
# In app.py

def get_openai_client():
    """
    Returns a custom OpenAI client implementation with the same interface
    but different internal authentication and routing.
    """
    # Import your custom client implementation
    from your_custom_module import CustomOpenAIClient
    
    # Return your custom client with the same interface
    return CustomOpenAIClient(
        api_key=config.openai_api_key,
        # Add your custom parameters here
        enterprise_id=config.enterprise_id,
        custom_routing=config.custom_routing
    )
```

### Example Use Cases

1. **Enterprise Routing**: Route requests to different OpenAI deployments based on business rules
2. **Custom Authentication**: Implement custom JWT, OAuth, or other auth schemes
3. **Request Transformation**: Modify requests before sending to OpenAI (e.g., adding context)
4. **Response Filtering**: Apply content filtering or modification to responses
5. **Logging & Analytics**: Add detailed logging for compliance or usage tracking

The examples directory contains a custom client implementation to demonstrate how to maintain the OpenAI API interface while implementing your own authentication and routing logic.

## Health and Monitoring

- Health check: `GET /health`
- API metrics: `GET /metrics`

## Testing

Run the test suite:

```bash
pytest
```
