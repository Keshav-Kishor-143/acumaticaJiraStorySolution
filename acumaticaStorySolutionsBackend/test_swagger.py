"""
Quick test script to verify Swagger UI is accessible
"""
import requests
import json

BASE_URL = "http://localhost:8001"

def test_swagger_endpoints():
    """Test Swagger-related endpoints"""
    print("=" * 60)
    print("Testing Swagger UI Configuration")
    print("=" * 60)
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/docs", "Swagger UI"),
        ("/redoc", "ReDoc documentation"),
        ("/openapi.json", "OpenAPI Schema"),
        ("/health", "Health check"),
        ("/solutions/health", "Solutions health check")
    ]
    
    for endpoint, description in endpoints:
        try:
            url = f"{BASE_URL}{endpoint}"
            print(f"\n📡 Testing: {description}")
            print(f"   URL: {url}")
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                if endpoint == "/openapi.json":
                    data = response.json()
                    print(f"   📄 OpenAPI Version: {data.get('openapi', 'N/A')}")
                    print(f"   📋 Title: {data.get('info', {}).get('title', 'N/A')}")
                    print(f"   🔌 Endpoints: {len(data.get('paths', {}))}")
            else:
                print(f"   ⚠️  Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error: Service not running!")
            print(f"   💡 Start the service with: python main.py")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Swagger UI Access:")
    print(f"   🌐 Swagger UI: {BASE_URL}/docs")
    print(f"   📚 ReDoc: {BASE_URL}/redoc")
    print(f"   📋 OpenAPI Schema: {BASE_URL}/openapi.json")
    print("=" * 60)

if __name__ == "__main__":
    test_swagger_endpoints()

