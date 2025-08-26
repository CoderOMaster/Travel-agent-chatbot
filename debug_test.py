import requests
import json

# Test the API endpoints
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat_validation():
    """Test chat endpoint with various payloads"""
    
    # Test valid payload
    valid_payload = {
        "message": "Hello, I need help with hotels",
        "session_id": None
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat", 
            json=valid_payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Valid payload test: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Valid payload test failed: {e}")
    
    # Test invalid payloads
    invalid_payloads = [
        {},  # Empty
        {"message": "ast to chd"},  # Empty message
        {"message": None},  # None message
        {"wrong_field": "test"}  # Wrong field
    ]
    
    for i, payload in enumerate(invalid_payloads):
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat", 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"Invalid payload {i+1}: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Invalid payload {i+1} test failed: {e}")

def test_streaming():
    """Test streaming endpoint"""
    payload = {
        "message": "Find me hotels in Paris",
        "session_id": None
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/stream",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            },
            stream=True
        )
        
        print(f"Streaming test: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("Streaming content:")
            for line in response.iter_lines():
                if line:
                    print(f"Received: {line.decode()}")
                    # Stop after a few lines to avoid long output
                    break
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Streaming test failed: {e}")

if __name__ == "__main__":
    print("=== API Debug Tests ===")
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health()
    
    if health_ok:
        print("\n2. Testing Chat Validation...")
        test_chat_validation()
        
        print("\n3. Testing Streaming...")
        test_streaming()
    else:
        print("Server is not running or not accessible")
        print("Make sure to start the server with: python app.py")