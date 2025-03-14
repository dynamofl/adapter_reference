from fastapi.testclient import TestClient
import json
import os
import sys

# Import app from the main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app

# Create test client
client = TestClient(app)

def test_health():
    response = client.get("/health")
    print(f"Health check status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_root():
    response = client.get("/")
    print(f"Root endpoint status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_validation_error():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": []  # Empty messages array should fail validation
        }
    )
    print(f"Validation error test status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 422 and "error" in response.json()

def test_role_validation():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "invalid_role", "content": "This should fail validation"}
            ]
        }
    )
    print(f"Role validation test status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 422 and "error" in response.json()

def test_auth_requirement():
    # This should pass when ENABLE_AUTH is False (default)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
    )
    print(f"Auth test status: {response.status_code}")
    print(f"Status is not 401: {response.status_code != 401}")
    return response.status_code != 401  # Should not be unauthorized

def test_metrics_endpoint():
    response = client.get("/metrics")
    print(f"Metrics endpoint status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

if __name__ == "__main__":
    print("Testing OpenAI Proxy API...")
    
    # Run tests
    health_result = test_health()
    print(f"Health test passed: {health_result}")
    print("\n---\n")
    
    root_result = test_root()
    print(f"Root test passed: {root_result}")
    print("\n---\n")
    
    validation_result = test_validation_error()
    print(f"Validation error test passed: {validation_result}")
    print("\n---\n")
    
    role_validation_result = test_role_validation()
    print(f"Role validation test passed: {role_validation_result}")
    print("\n---\n")
    
    auth_result = test_auth_requirement()
    print(f"Auth test passed: {auth_result}")
    print("\n---\n")
    
    metrics_result = test_metrics_endpoint()
    print(f"Metrics test passed: {metrics_result}")
    
    # Overall result
    all_passed = (
        health_result and 
        root_result and 
        validation_result and 
        role_validation_result and 
        auth_result and 
        metrics_result
    )
    print("\n===\n")
    print(f"All tests passed: {all_passed}")
    
    # Exit with appropriate exit code
    sys.exit(0 if all_passed else 1)