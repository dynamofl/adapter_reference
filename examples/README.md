# Custom Adapter Examples

This directory contains example adapters showing how to extend the OpenAI API Proxy to work with different AI providers.

## Available Adapters

### Anthropic Claude Adapter

The `anthropic_adapter.py` example demonstrates how to create an OpenAI-compatible API that uses Anthropic's Claude models behind the scenes. This allows you to use OpenAI client libraries to interact with Claude.

### Mistral AI Adapter

The `mistral_adapter.py` example shows how to create an adapter for Mistral's API, allowing you to use the OpenAI client with Mistral models.

## Creating Your Own Adapter

To create a custom adapter:

1. Start by copying one of the example adapters
2. Update the configuration to include your API provider's settings
3. Create mapping functions to convert between OpenAI format and your provider's format
4. Implement the endpoints you need (usually just `/v1/chat/completions`)

## Best Practices

When creating an adapter, consider the following best practices:

1. **Model Mapping**: Create a clear mapping between OpenAI model names and your provider's model names
2. **Error Handling**: Convert your provider's error messages to OpenAI-compatible formats
3. **Streaming Support**: Implement streaming properly to maintain compatibility
4. **Authentication**: Reuse the authentication system from the base proxy
5. **Documentation**: Document how to use your adapter, including any limitations or differences