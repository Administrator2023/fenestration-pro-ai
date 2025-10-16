"""
BQE Core OAuth 2.0 Integration Helper
"""
import streamlit as st
import requests
import json
import base64
import secrets
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta

class BQEOAuth:
    """Handle BQE Core OAuth 2.0 authentication flow"""
    
    # OAuth endpoints
    AUTHORIZE_URL = "https://api.bqecore.com/connect/authorize"
    TOKEN_URL = "https://api.bqecore.com/connect/token"
    
    def __init__(self, client_id, redirect_uri, base_api_url="https://api.bqecore.com/api"):
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.base_api_url = base_api_url
        
    def get_authorization_url(self, state=None):
        """Generate the OAuth authorization URL"""
        if not state:
            state = secrets.token_urlsafe(32)
            
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid profile email offline_access api",
            "state": state
        }
        
        auth_url = f"{self.AUTHORIZE_URL}?{urlencode(params)}"
        return auth_url, state
    
    def exchange_code_for_token(self, code, client_secret):
        """Exchange authorization code for access token"""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": client_secret
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(self.TOKEN_URL, data=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Token exchange failed: {response.status_code} - {response.text}")
    
    def refresh_token(self, refresh_token, client_secret):
        """Refresh an expired access token"""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": client_secret
        }
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        response = requests.post(self.TOKEN_URL, data=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Token refresh failed: {response.status_code} - {response.text}")
    
    def make_api_request(self, endpoint, access_token, method="GET", data=None):
        """Make an authenticated API request"""
        url = f"{self.base_api_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response
    
def init_oauth_session():
    """Initialize OAuth session state"""
    if "bqe_oauth_state" not in st.session_state:
        st.session_state.bqe_oauth_state = None
    if "bqe_access_token" not in st.session_state:
        st.session_state.bqe_access_token = None
    if "bqe_refresh_token" not in st.session_state:
        st.session_state.bqe_refresh_token = None
    if "bqe_token_expires" not in st.session_state:
        st.session_state.bqe_token_expires = None
    if "bqe_client_id" not in st.session_state:
        st.session_state.bqe_client_id = "U2pwazJCTFbCq7Re6VkR31YQc48pcL_O.apps.bqe.com"
    if "bqe_client_secret" not in st.session_state:
        st.session_state.bqe_client_secret = ""

def handle_oauth_callback():
    """Handle OAuth callback with authorization code"""
    query_params = st.experimental_get_query_params()
    
    if "code" in query_params and "state" in query_params:
        code = query_params["code"][0]
        state = query_params["state"][0]
        
        # Verify state matches
        if state == st.session_state.get("bqe_oauth_state"):
            return code
        else:
            st.error("OAuth state mismatch. Please try logging in again.")
            return None
    
    return None

def is_token_valid():
    """Check if the current token is still valid"""
    if not st.session_state.bqe_access_token:
        return False
    
    if st.session_state.bqe_token_expires:
        return datetime.now() < st.session_state.bqe_token_expires
    
    return False