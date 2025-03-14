# OpenAI Adapter

An API proxy service for OpenAI that provides authentication, error handling, and request validation for chat completions endpoints.

## Features

- **Focused API**: Supports only the core completions endpoints
- **Authentication**: Optional API key authentication for securing your proxy
- **Request Validation**: Validates all requests before sending to OpenAI
- **Improved Error Handling**: Returns appropriate HTTP status codes and sanitized error messages
- **Metrics**: Built-in metrics endpoint for monitoring
- **Streaming Support**: Fully supports streaming responses for chat and completions
- **Health Check**: Built-in health check endpoint for monitoring

## Supported Endpoints

- `/v1/chat/completions` - Chat completions API
- `/v1/completions` - Text completions API

Note: Other OpenAI endpoints like embeddings, image generation, and model listing are not supported in this focused adapter.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- An OpenAI API key

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

4. Edit the `.env` file and add your OpenAI API key:
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
| `ENABLE_AUTH` | Enable API key authentication | `false` |
| `API_KEY_SECRET` | API key for authentication when enabled | Random generated |
| `PORT` | Port to run the server on | `8000` |

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

## Health and Monitoring

- Health check: `GET /health`
- API metrics: `GET /metrics` (protected by API key when `ENABLE_AUTH=true`)

## Testing

Run the test suite:

```bash
pytest
```