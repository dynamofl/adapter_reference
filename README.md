# OpenAI API Proxy/Adapter

A minimal API proxy service for OpenAI that focuses exclusively on text generation endpoints.

## Features

- **Focused API Compatibility**: Supports only the core text generation endpoints
- **Authentication**: Optional API key authentication for securing your proxy
- **Request Validation**: Validates all requests before sending to OpenAI
- **Metrics**: Basic metrics endpoint for monitoring
- **Error Handling**: Improved error handling with sanitized messages
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
git clone https://github.com/your-username/openai-proxy.git
cd openai-proxy
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

The following environment variables can be used to configure the proxy:

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

# Embeddings
embeddings_response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Hello, world!"
)
print(embeddings_response.data[0].embedding)

# Image generation
image_response = client.images.generate(
    prompt="A beautiful sunset over mountains",
    n=1,
    size="1024x1024"
)
print(image_response.data[0].url)
```

## Health and Monitoring

- Health check: `GET /health`
- API metrics: `GET /metrics` (protected by API key when `ENABLE_AUTH=true`)

## Testing

Run the test suite:

```bash
pytest
```

With coverage:

```bash
coverage run -m pytest
coverage report
```

## Development

Format code:

```bash
black .
```

Lint code:

```bash
ruff check .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

