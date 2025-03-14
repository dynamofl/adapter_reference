#!/usr/bin/env python3
"""
End-to-end tests for the OpenAI adapter.

This script tests the actual running API service by making real requests to the endpoints.
It validates both successful responses and error handling for various scenarios.

Usage:
    python -m tests.e2e_tests [--base-url BASE_URL] [--api-key API_KEY]

Options:
    --base-url BASE_URL    Base URL of the API (default: http://localhost:8000)
    --api-key API_KEY      API key for authentication if ENABLE_AUTH is true
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List, Optional, Union
import requests
from requests.exceptions import RequestException

# Global variables
args = None
tests_run = 0
tests_passed = 0
tests_failed = 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="End-to-end tests for the OpenAI adapter")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication if ENABLE_AUTH is true"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including full API responses"
    )
    return parser.parse_args()


def get_headers():
    """Get headers for API requests"""
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    return headers


def print_separator():
    """Print a separator line"""
    print("\n" + "=" * 80)


def run_test(name: str, func):
    """Run a test function and track results"""
    global tests_run, tests_passed, tests_failed
    
    print_separator()
    print(f"RUNNING TEST: {name}")
    print("-" * 80)
    
    tests_run += 1
    start_time = time.time()
    
    try:
        result = func()
        if result:
            tests_passed += 1
            elapsed = time.time() - start_time
            print(f"\n✅ PASSED: {name} ({elapsed:.2f}s)")
            return True
        else:
            tests_failed += 1
            elapsed = time.time() - start_time
            print(f"\n❌ FAILED: {name} ({elapsed:.2f}s)")
            return False
    except Exception as e:
        tests_failed += 1
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR: {name} - {str(e)} ({elapsed:.2f}s)")
        return False


def print_response(response, max_length=500):
    """Print a formatted response"""
    print(f"Status Code: {response.status_code}")
    print("Headers:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print("\nResponse:")
    try:
        json_data = response.json()
        json_text = json.dumps(json_data, indent=2)
        if not args.verbose and len(json_text) > max_length:
            print(json_text[:max_length] + "...\n(truncated, use --verbose for full response)")
        else:
            print(json_text)
    except:
        text = response.text
        if not args.verbose and len(text) > max_length:
            print(text[:max_length] + "...\n(truncated, use --verbose for full response)")
        else:
            print(text)


def assert_response(response, expected_status=200, ensure_json=True, 
                    expected_keys=None, error_contains=None):
    """Assert that a response meets expectations"""
    
    if response.status_code != expected_status:
        print(f"❌ Expected status code {expected_status}, got {response.status_code}")
        print_response(response)
        return False
    
    if ensure_json:
        try:
            json_data = response.json()
        except:
            print(f"❌ Expected JSON response, got: {response.text[:200]}")
            return False
    
    if expected_keys:
        try:
            json_data = response.json()
            for key in expected_keys:
                if key not in json_data:
                    print(f"❌ Expected key '{key}' not found in response")
                    print_response(response)
                    return False
        except:
            print(f"❌ Could not check for keys in non-JSON response")
            return False
    
    if error_contains and ensure_json:
        try:
            json_data = response.json()
            if "error" in json_data:
                error_message = json_data.get("detail", "")
                if isinstance(error_message, dict) and "detail" in error_message:
                    error_message = error_message["detail"]
                
                if error_contains not in str(error_message) and error_contains not in json_data.get("error", ""):
                    print(f"❌ Expected error containing '{error_contains}', got: {error_message}")
                    print_response(response)
                    return False
            else:
                print(f"❌ Expected error response, got: {json_data}")
                return False
        except:
            print(f"❌ Could not check for error in non-JSON response")
            return False
    
    return True


def print_summary():
    """Print a summary of test results"""
    print_separator()
    print("TEST SUMMARY")
    print("-" * 80)
    print(f"Total tests: {tests_run}")
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print_separator()
    
    if tests_failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {tests_failed} test(s) failed.")
    
    return tests_failed == 0


# TESTS
def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    
    response = requests.get(f"{args.base_url}/health")
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["status", "version", "timestamp"]
    )


def test_root_endpoint():
    """Test the root endpoint"""
    print("Testing root endpoint...")
    
    response = requests.get(args.base_url)
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["message"]
    )


def test_metrics_endpoint():
    """Test the metrics endpoint"""
    print("Testing metrics endpoint...")
    
    response = requests.get(
        f"{args.base_url}/metrics",
        headers=get_headers()
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["uptime_seconds", "version"]
    )


def test_chat_completion_basic():
    """Test basic chat completion"""
    print("Testing basic chat completion...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model"]
    )


def test_chat_completion_with_system():
    """Test chat completion with system message"""
    print("Testing chat completion with system message...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
            {"role": "user", "content": "Tell me about the weather today."}
        ]
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model"]
    )


def test_chat_completion_with_parameters():
    """Test chat completion with various parameters"""
    print("Testing chat completion with parameters...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Write a short poem about programming."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "n": 1,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.3
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model", "usage"]
    )


def test_completions_basic():
    """Test basic completions (legacy) endpoint"""
    print("Testing basic completions endpoint...")
    
    payload = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }
    
    response = requests.post(
        f"{args.base_url}/v1/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model", "usage"]
    )


def test_completions_with_parameters():
    """Test completions with various parameters"""
    print("Testing completions with parameters...")
    
    payload = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Write a recipe for",
        "max_tokens": 50,
        "temperature": 0.5,
        "top_p": 0.8,
        "n": 1,
        "stop": ["\n\n"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2
    }
    
    response = requests.post(
        f"{args.base_url}/v1/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model", "usage"]
    )


def test_completions_with_array_prompt():
    """Test completions with array prompt"""
    print("Testing completions with array prompt...")
    
    payload = {
        "model": "gpt-3.5-turbo-instruct",
        "prompt": ["Hello, how are you?", "What is the weather like today?"],
        "max_tokens": 50
    }
    
    response = requests.post(
        f"{args.base_url}/v1/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=200, 
        expected_keys=["choices", "id", "model", "usage"]
    )


def test_chat_completion_streaming():
    """Test streaming chat completion"""
    print("Testing streaming chat completion...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Count from 1 to 5 briefly."}
        ],
        "stream": True
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload,
        stream=True
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', '')}")
    
    if response.status_code != 200:
        print(f"❌ Expected status code 200, got {response.status_code}")
        return False
    
    if "text/event-stream" not in response.headers.get("content-type", ""):
        print(f"❌ Expected content-type containing 'text/event-stream', got {response.headers.get('content-type', '')}")
        return False
    
    # Print the first few chunks to verify streaming
    print("\nResponse (first few chunks):")
    chunk_count = 0
    for chunk in response.iter_lines():
        if chunk:
            print(chunk.decode('utf-8'))
            chunk_count += 1
            if chunk_count >= 5:
                print("...\n(more chunks follow)")
                break
    
    return True


def test_validation_empty_messages():
    """Test validation for empty messages array"""
    print("Testing validation for empty messages array...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": []  # Empty messages array should fail validation
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=422, 
        error_contains="message"
    )


def test_validation_invalid_role():
    """Test validation for invalid message role"""
    print("Testing validation for invalid message role...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "invalid_role", "content": "This should fail validation"}
        ]
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=422, 
        error_contains="role"
    )


def test_validation_missing_required_field():
    """Test validation for missing required field"""
    print("Testing validation for missing required field...")
    
    # Missing model field
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=422, 
        error_contains="model"
    )


def test_validation_parameter_constraints():
    """Test validation for parameter constraints"""
    print("Testing validation for parameter constraints...")
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 3.0  # Outside valid range (0-2)
    }
    
    response = requests.post(
        f"{args.base_url}/v1/chat/completions",
        headers=get_headers(),
        json=payload
    )
    print_response(response)
    
    return assert_response(
        response, 
        expected_status=422, 
        error_contains="temperature"
    )


def test_unsupported_endpoints():
    """Test that unsupported endpoints return 404"""
    print("Testing unsupported endpoints...")
    
    # Test embeddings endpoint
    embeddings_payload = {
        "model": "text-embedding-ada-002",
        "input": "Hello world"
    }
    
    # Test models endpoint
    models_response = requests.get(
        f"{args.base_url}/v1/models",
        headers=get_headers()
    )
    
    # Test embeddings endpoint
    embeddings_response = requests.post(
        f"{args.base_url}/v1/embeddings",
        headers=get_headers(),
        json=embeddings_payload
    )
    
    # Test images endpoint
    images_payload = {
        "prompt": "A beautiful sunset",
        "n": 1
    }
    
    images_response = requests.post(
        f"{args.base_url}/v1/images/generations",
        headers=get_headers(),
        json=images_payload
    )
    
    print("\nModels endpoint:")
    print_response(models_response)
    
    print("\nEmbeddings endpoint:")
    print_response(embeddings_response)
    
    print("\nImages endpoint:")
    print_response(images_response)
    
    return (
        models_response.status_code == 404 and
        embeddings_response.status_code == 404 and
        images_response.status_code == 404
    )


def main():
    """Main function to run all tests"""
    global args
    args = parse_args()
    
    print(f"Running end-to-end tests against {args.base_url}")
    if args.api_key:
        print("Using provided API key for authentication")
    
    try:
        # Basic endpoints
        run_test("Health Check", test_health_check)
        run_test("Root Endpoint", test_root_endpoint)
        run_test("Metrics Endpoint", test_metrics_endpoint)
        
        # Chat completions
        run_test("Basic Chat Completion", test_chat_completion_basic)
        run_test("Chat Completion with System Message", test_chat_completion_with_system)
        run_test("Chat Completion with Parameters", test_chat_completion_with_parameters)
        
        # Completions
        run_test("Basic Completions", test_completions_basic)
        run_test("Completions with Parameters", test_completions_with_parameters)
        run_test("Completions with Array Prompt", test_completions_with_array_prompt)
        
        # Streaming
        run_test("Streaming Chat Completion", test_chat_completion_streaming)
        
        # Validation
        run_test("Validation: Empty Messages", test_validation_empty_messages)
        run_test("Validation: Invalid Role", test_validation_invalid_role)
        run_test("Validation: Missing Required Field", test_validation_missing_required_field)
        run_test("Validation: Parameter Constraints", test_validation_parameter_constraints)
        
        # Unsupported endpoints
        run_test("Unsupported Endpoints", test_unsupported_endpoints)
        
        # Print summary
        success = print_summary()
        sys.exit(0 if success else 1)
        
    except RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the API at {args.base_url}")
        print(f"    {str(e)}")
        print("\nPlease make sure the API server is running and accessible.")
        sys.exit(1)


if __name__ == "__main__":
    main()