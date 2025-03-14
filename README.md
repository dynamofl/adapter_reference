# OpenAI Adapter

A flexible API proxy service for OpenAI and other AI providers. This adapter provides authentication, error handling, request validation, and allows you to use OpenAI-compatible clients with other AI services.

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
- **Extensibility**: Example adapters for other AI providers (Anthropic, Mistral)

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

### Authentication

When authentication is enabled, include your API key in the header:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
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

### Using with the OpenAI Python client

```python
import openai

client = openai.OpenAI(
    # No API key needed when ENABLE_AUTH=false
    base_url="http://localhost:8000/v1",
    # Include API key in header when ENABLE_AUTH=true
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

## Creating Custom Adapters

The adapter is designed to be extended to work with different AI providers. Check the `examples/` directory for adapter examples:

### Anthropic Claude Adapter

This adapter allows you to use the OpenAI client with Anthropic's Claude models.

1. Install the Anthropic Python SDK:
```bash
pip install anthropic
```

2. Add your Anthropic API key to `.env`:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

3. Run the adapter:
```bash
python examples/anthropic_adapter.py
```

Now you can use the OpenAI client with Claude models:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
)

# This will actually use Claude even though you specify an OpenAI model name
response = client.chat.completions.create(
    model="gpt-4",  # Will be mapped to claude-3-opus-20240229
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)
print(response.choices[0].message.content)
```

### Mistral AI Adapter

This adapter allows you to use the OpenAI client with Mistral's models.

1. Install the Mistral Python SDK:
```bash
pip install mistralai
```

2. Add your Mistral API key to `.env`:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

3. Run the adapter:
```bash
python examples/mistral_adapter.py
```

### Creating Your Own Adapter

To create your own adapter:

1. Copy one of the example adapters as a starting point
2. Update the configuration to connect to your API provider
3. Create mappings between OpenAI request format and your provider's format
4. Implement the endpoints needed for your use case

## Health and Monitoring

- Health check: `GET /health`
- API metrics: `GET /metrics` (protected by API key when `ENABLE_AUTH=true`)

## Testing

Run the test suite:

```bash
pytest
```