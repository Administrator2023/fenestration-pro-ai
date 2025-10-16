#!/usr/bin/env python3
"""Test different BQE OAuth endpoint possibilities"""
import requests

# Based on client ID: U2pwazJCTFbCq7Re6VkR31YQc48pcL_O.apps.bqe.com
possible_oauth_urls = [
    # BQE Core variants
    "https://api.bqecore.com/oauth/authorize",
    "https://api.bqecore.com/identity/connect/authorize",
    "https://api.bqecore.com/connect/authorize",
    
    # apps.bqe.com variants (based on client ID domain)
    "https://apps.bqe.com/oauth/authorize",
    "https://apps.bqe.com/identity/connect/authorize",
    "https://apps.bqe.com/connect/authorize",
    
    # Other possibilities
    "https://auth.bqe.com/oauth/authorize",
    "https://login.bqe.com/oauth/authorize",
    "https://oauth.bqe.com/authorize",
    "https://identity.bqe.com/connect/authorize",
    
    # BQE CORE specific
    "https://bqecore.com/oauth/authorize",
    "https://secure.bqecore.com/oauth/authorize",
]

print("Testing BQE OAuth endpoints...")
print("=" * 60)

for url in possible_oauth_urls:
    try:
        # Just check if the endpoint exists (HEAD request)
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code < 500:  # Anything less than 500 means the endpoint exists
            print(f"✓ {url} - Status: {response.status_code}")
        else:
            print(f"✗ {url} - Server Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"✗ {url} - Connection failed")
    except requests.exceptions.Timeout:
        print(f"✗ {url} - Timeout")
    except Exception as e:
        print(f"✗ {url} - Error: {str(e)}")

print("\n" + "=" * 60)
print("Note: Even 404 or 405 means the server exists and might have OAuth at a different path")