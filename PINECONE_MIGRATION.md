# Pinecone Package Migration Guide

## Issue

The official Pinecone Python package has been renamed from `pinecone-client` to `pinecone`. If you have both packages installed, you will see this error:

```
Exception: The official Pinecone python package has been renamed from `pinecone-client` to `pinecone`.
Please remove `pinecone-client` from your project dependencies and add `pinecone` instead.
```

## Solution

You **MUST** uninstall the old `pinecone-client` package before installing the new `pinecone` package.

### Step 1: Uninstall the old package

```bash
pip uninstall pinecone-client
```

### Step 2: Install the new package

```bash
pip install pinecone>=3.0.0
```

Or simply reinstall all requirements:

```bash
pip install -r requirements.txt
```

### For Streamlit Cloud Users

If you're deploying on Streamlit Cloud:

1. Make sure your `requirements.txt` only lists `pinecone>=3.0.0` (not `pinecone-client`)
2. **Reboot your app** from the Streamlit Cloud dashboard to force a clean reinstall
3. If the issue persists, click "Manage app" → "Settings" → "Clear cache" → Reboot

### Verification

After installation, verify the package is correctly installed:

```bash
pip list | grep pinecone
```

You should see:
```
pinecone            3.x.x
```

You should **NOT** see `pinecone-client` in the list.

## Code Changes

The import statement remains the same:

```python
from pinecone import Pinecone, ServerlessSpec
```

No code changes are required, only the package name in `requirements.txt` has changed.
