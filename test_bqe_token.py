#!/usr/bin/env python3
"""Test different ways to use the BQE token"""
import requests
import base64

token = "qiXSQ2uKoeF9b5M7bOKtRYNpBxBaVw1c955M0fFU_ldZ2cjovtMSlkbT28aJaBPl"
base_url = "https://api.bqecore.com/api"

print("Testing BQE Core API Token...")
print("=" * 60)

# Test 1: Bearer token (what we're currently using)
print("\nTest 1: Bearer Token")
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}
response = requests.get(f"{base_url}/employee", headers=headers)
print(f"Status: {response.status_code}")
print(f"Headers: {dict(response.headers)}")
if response.text:
    print(f"Response: {response.text[:200]}")

# Test 2: Basic Auth (token as username)
print("\n\nTest 2: Basic Auth (token as username)")
headers = {
    "Authorization": f"Basic {base64.b64encode(f'{token}:'.encode()).decode()}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}
response = requests.get(f"{base_url}/employee", headers=headers)
print(f"Status: {response.status_code}")

# Test 3: Token in header without Bearer
print("\n\nTest 3: Token without Bearer prefix")
headers = {
    "Authorization": token,
    "Accept": "application/json",
    "Content-Type": "application/json"
}
response = requests.get(f"{base_url}/employee", headers=headers)
print(f"Status: {response.status_code}")

# Test 4: X-API-Key header
print("\n\nTest 4: X-API-Key header")
headers = {
    "X-API-Key": token,
    "Accept": "application/json",
    "Content-Type": "application/json"
}
response = requests.get(f"{base_url}/employee", headers=headers)
print(f"Status: {response.status_code}")

# Test 5: Check if token should be in query params
print("\n\nTest 5: Token in query params")
response = requests.get(f"{base_url}/employee?api_key={token}", headers={"Accept": "application/json"})
print(f"Status: {response.status_code}")

# Test 6: Try a different endpoint
print("\n\nTest 6: Try /account endpoint")
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
}
response = requests.get(f"{base_url}/account", headers=headers)
print(f"Status: {response.status_code}")

# Test 7: Try the base URL without /api
print("\n\nTest 7: Try without /api in URL")
response = requests.get("https://api.bqecore.com/employee", headers={
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
})
print(f"Status: {response.status_code}")

print("\n" + "=" * 60)
print("If all tests return 401, the token might be:")
print("1. Expired")
print("2. For a different environment (sandbox vs production)")
print("3. Missing required permissions")
print("4. Not yet activated")