# Fenestration Pro AI - Deployment Guide

## ✅ BQE Integration Complete

The BQE integration has been successfully added to the Fenestration Pro AI application.

### What's New:

1. **BQE Tab** in Project Management section with:
   - Base URL configuration (default: https://api.bqe.com/v1)
   - API Token input (secure/password field)
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

1. Navigate to the deployed app
2. Go to "Project Management" section
3. Click on the "BQE" tab
4. Enter your BQE credentials:
   - Keep the default Base URL or update if needed
   - Enter your BQE API Token
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

### API Endpoints:

The integration is configured for standard BQE API v1 endpoints:
- Account: `/account`
- Projects: `/projects`
- Contacts: `/contacts`

If your BQE instance uses different endpoints, you can modify the Base URL in the app.

---

**Last Updated**: October 16, 2025
**Branch**: main
**Status**: Ready for deployment