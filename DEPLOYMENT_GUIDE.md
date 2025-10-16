# Fenestration Pro AI - Deployment Guide

## ✅ BQE Integration Complete

The BQE integration has been successfully added to the Fenestration Pro AI application.

### What's New:

1. **BQE Core Tab** in Project Management section with:
   - **Dual Authentication Methods**:
     - OAuth 2.0 (Recommended) - Secure OAuth flow with refresh tokens
     - API Token - Direct token authentication
   - Base URL configuration (default: https://api.bqecore.com/api)
   - Test Connection functionality
   - Import Projects & Contacts feature
   - BQE sync status display

2. **Updated Dependencies**:
   - Added `requests` library to requirements.txt

3. **Code Changes**:
   - Modified `app.py` to include BQE integration tab
   - All changes merged to main branch

### To Deploy to Streamlit Cloud:

1. **Automatic Deployment** (if configured):
   - The app should automatically redeploy when changes are pushed to main branch
   - This typically takes 2-5 minutes

2. **Manual Deployment**:
   - Log into [Streamlit Cloud](https://share.streamlit.io)
   - Navigate to your app
   - Click "Manage app" → "Reboot app"

3. **Configuration in Streamlit Cloud**:
   - Go to app settings
   - Add secrets in the "Secrets" section:
     ```toml
     OPENAI_API_KEY = "your-openai-api-key"
     ADMIN_PASSWORD = "your-admin-password"
     ```

### Using the BQE Integration:

#### Option 1: OAuth 2.0 Authentication (Recommended)
1. Navigate to the deployed app
2. Go to "Project Management" → "BQE" tab
3. Select "OAuth 2.0 (Recommended)"
4. Enter your OAuth Client Secret
5. Click "Connect with BQE"
6. Authorize the app in BQE Core
7. You'll be redirected back and authenticated
8. Click "Import Projects & Contacts" to sync data

#### Option 2: API Token Authentication
1. Navigate to the deployed app
2. Go to "Project Management" → "BQE" tab
3. Select "API Token"
4. Enter your BQE Core API Token
5. Click "Test Connection" to verify
6. Click "Import Projects & Contacts" to sync data

### Features:
- Projects imported from BQE create new project folders
- Contacts are imported to the current project
- All imported data is tagged with `imported_from_bqe` flag
- Sync status shows when data was last imported

### Troubleshooting:

If the app doesn't update:
1. Check Streamlit Cloud logs for any errors
2. Ensure the GitHub webhook is connected
3. Try manual reboot from the dashboard
4. Verify all dependencies are compatible

### BQE Core OAuth Application:

The app is configured with:
- **Client ID**: `U2pwazJCTFbCq7Re6VkR31YQc48pcL_O.apps.bqe.com`
- **Redirect URI**: `https://fenestrationpro.streamlit.app/`
- **Scopes**: openid, profile, email, offline_access, api

### API Endpoints:

The integration uses BQE Core API v1 endpoints:
- Employee: `/employee` (for testing connection)
- Projects: `/project`
- Contacts: `/contact`

### Required Secrets:

For OAuth, you'll need to add your Client Secret in the app or Streamlit Cloud secrets:
```toml
BQE_CLIENT_SECRET = "your-oauth-client-secret"
```

---

**Last Updated**: October 16, 2025
**Branch**: main
**Status**: Ready for deployment