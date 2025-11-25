# OCR Provider Credentials Setup Guide

This guide explains how to set up credentials for each OCR provider.

---

## Table of Contents

1. [Google Cloud Vision](#1-google-cloud-vision)
2. [Gemini Vision (Google AI)](#2-gemini-vision-google-ai)
3. [Claude Vision (Anthropic)](#3-claude-vision-anthropic)
4. [OpenAI Vision (GPT-4o)](#4-openai-vision-gpt-4o)
5. [Azure Document Intelligence](#5-azure-document-intelligence)
6. [AWS Textract](#6-aws-textract)
7. [Local Providers (No API Key)](#7-local-providers-no-api-key)

---

## 1. Google Cloud Vision

**Pricing:** ~$1.50 per 1,000 images

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" > "New Project"
3. Name your project and click "Create"

### Step 2: Enable the Vision API
1. In Google Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Cloud Vision API"
3. Click "Enable"

### Step 3: Create Service Account Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "Service Account"
3. Name the service account (e.g., "ocr-service")
4. Click "Create and Continue"
5. Grant role: "Cloud Vision API User"
6. Click "Done"

### Step 4: Download JSON Key
1. Click on the service account you created
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select "JSON" and click "Create"
5. Save the downloaded JSON file securely

### Step 5: Set Up Credentials
**Option A: Environment Variable**
```bash
# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your-key.json"

# Windows (CMD)
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-key.json

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
```

**Option B: Keyring (Recommended)**
```bash
# Store the entire JSON content
python -c "import keyring; keyring.set_password('google_cloud', 'service_account_json', open('path/to/key.json').read())"
```

### Verify Setup
```bash
python -m ocr.providers.google_vision check
```

---

## 2. Gemini Vision (Google AI)

**Pricing:** ~$0.001 per image (very cheap!)

### Step 1: Get API Key
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

### Step 2: Set Up Credentials
**Option A: Keyring (Recommended - shares with TTS)**
```bash
python -c "import keyring; keyring.set_password('gemini', 'api_key', 'YOUR_API_KEY_HERE')"
```

**Option B: Environment Variable**
```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# Windows (CMD)
set GEMINI_API_KEY=YOUR_API_KEY_HERE

# Linux/Mac
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### Verify Setup
```bash
python -m ocr.providers.gemini_vision check
```

---

## 3. Claude Vision (Anthropic)

**Pricing:** ~$0.0004/image (Haiku) to ~$0.005/image (Sonnet)

### Step 1: Get API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to "API Keys"
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

### Step 2: Set Up Credentials
**Option A: Keyring (Recommended)**
```bash
python -c "import keyring; keyring.set_password('anthropic', 'api_key', 'sk-ant-YOUR_KEY_HERE')"
```

**Option B: Environment Variable**
```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-YOUR_KEY_HERE"

# Windows (CMD)
set ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE

# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-YOUR_KEY_HERE"
```

### Verify Setup
```bash
python -m ocr.providers.claude_vision check
```

---

## 4. OpenAI Vision (GPT-4o)

**Pricing:** ~$0.001/image (4o-mini) to ~$0.008/image (4o)

### Step 1: Get API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Go to "API Keys" (left sidebar)
4. Click "Create new secret key"
5. Name it and copy the key (starts with `sk-`)

### Step 2: Add Credits
1. Go to "Billing" > "Payment methods"
2. Add a payment method
3. Add credits to your account

### Step 3: Set Up Credentials
**Option A: Keyring (Recommended)**
```bash
python -c "import keyring; keyring.set_password('openai', 'api_key', 'sk-YOUR_KEY_HERE')"
```

**Option B: Environment Variable**
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-YOUR_KEY_HERE"

# Windows (CMD)
set OPENAI_API_KEY=sk-YOUR_KEY_HERE

# Linux/Mac
export OPENAI_API_KEY="sk-YOUR_KEY_HERE"
```

### Verify Setup
```bash
python -m ocr.providers.openai_vision check
```

---

## 5. Azure Document Intelligence

**Pricing:** ~$1.50 per 1,000 pages (Read API)

### Step 1: Create Azure Account
1. Go to [Azure Portal](https://portal.azure.com/)
2. Sign up for a free account (includes $200 credit)

### Step 2: Create Document Intelligence Resource
1. Click "Create a resource"
2. Search for "Document Intelligence"
3. Click "Create"
4. Fill in:
   - Subscription: Your subscription
   - Resource group: Create new or use existing
   - Region: Choose nearest
   - Name: Unique name (e.g., "my-doc-intel")
   - Pricing tier: F0 (free) or S0 (standard)
5. Click "Review + create" > "Create"

### Step 3: Get Endpoint and Key
1. Go to your Document Intelligence resource
2. Click "Keys and Endpoint" (left sidebar)
3. Copy:
   - **Endpoint**: Something like `https://your-name.cognitiveservices.azure.com/`
   - **Key 1**: Your API key

### Step 4: Set Up Credentials
**Option A: Keyring (Recommended)**
```bash
python -c "import keyring; keyring.set_password('azure_doc_intel', 'endpoint', 'https://your-name.cognitiveservices.azure.com/')"
python -c "import keyring; keyring.set_password('azure_doc_intel', 'api_key', 'YOUR_KEY_HERE')"
```

**Option B: Environment Variables**
```bash
# Windows (PowerShell)
$env:AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = "https://your-name.cognitiveservices.azure.com/"
$env:AZURE_DOCUMENT_INTELLIGENCE_KEY = "YOUR_KEY_HERE"

# Windows (CMD)
set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-name.cognitiveservices.azure.com/
set AZURE_DOCUMENT_INTELLIGENCE_KEY=YOUR_KEY_HERE

# Linux/Mac
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://your-name.cognitiveservices.azure.com/"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="YOUR_KEY_HERE"
```

### Verify Setup
```bash
python -m ocr.providers.azure_doc_intel check
```

---

## 6. AWS Textract

**Pricing:** ~$1.50 per 1,000 pages

### Step 1: Create AWS Account
1. Go to [AWS Console](https://aws.amazon.com/)
2. Click "Create an AWS Account"
3. Complete signup (requires credit card)

### Step 2: Create IAM User
1. Go to IAM service in AWS Console
2. Click "Users" > "Add users"
3. User name: e.g., "textract-user"
4. Click "Next"
5. Select "Attach policies directly"
6. Search and select: "AmazonTextractFullAccess"
7. Click "Next" > "Create user"

### Step 3: Create Access Key
1. Click on your new user
2. Go to "Security credentials" tab
3. Under "Access keys", click "Create access key"
4. Select "Application running outside AWS"
5. Click "Next" > "Create access key"
6. **IMPORTANT**: Copy both:
   - Access key ID
   - Secret access key (only shown once!)

### Step 4: Set Up Credentials
**Option A: AWS CLI (Recommended)**
```bash
# Install AWS CLI first
pip install awscli

# Configure
aws configure
# Enter:
#   AWS Access Key ID: YOUR_ACCESS_KEY
#   AWS Secret Access Key: YOUR_SECRET_KEY
#   Default region name: us-east-1 (or your preferred region)
#   Default output format: json
```

**Option B: Environment Variables**
```bash
# Windows (PowerShell)
$env:AWS_ACCESS_KEY_ID = "YOUR_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_SECRET_KEY"
$env:AWS_DEFAULT_REGION = "us-east-1"

# Windows (CMD)
set AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
set AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
set AWS_DEFAULT_REGION=us-east-1

# Linux/Mac
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="us-east-1"
```

### Verify Setup
```bash
python -m ocr.providers.aws_textract check
```

---

## 7. Local Providers (No API Key)

These providers run locally and don't need API credentials:

### Tesseract
```bash
# Install Python package
pip install pytesseract pillow

# Install Tesseract OCR engine
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Mac: brew install tesseract
# Linux: sudo apt install tesseract-ocr

# Verify
python -m ocr.providers.tesseract_provider check
```

### EasyOCR
```bash
pip install easyocr
python -m ocr.providers.easyocr_provider check
```

### PaddleOCR
```bash
pip install paddleocr paddlepaddle
python -m ocr.providers.paddleocr_provider check
```

### docTR
```bash
pip install python-doctr[torch]
python -m ocr.providers.doctr_provider check
```

### Windows OCR (Windows only)
```bash
pip install winocr
python -m ocr.providers.windows_ocr check
```

---

## Quick Reference: All Credentials in One Script

Run this Python script to set up all your keyring credentials at once:

```python
import keyring

# Replace with your actual keys
credentials = {
    ('gemini', 'api_key'): 'YOUR_GEMINI_KEY',
    ('anthropic', 'api_key'): 'YOUR_ANTHROPIC_KEY',
    ('openai', 'api_key'): 'YOUR_OPENAI_KEY',
    ('azure_doc_intel', 'endpoint'): 'YOUR_AZURE_ENDPOINT',
    ('azure_doc_intel', 'api_key'): 'YOUR_AZURE_KEY',
}

for (service, key), value in credentials.items():
    if value and not value.startswith('YOUR_'):
        keyring.set_password(service, key, value)
        print(f"Set {service}/{key}")
```

---

## Verify All Providers

Run this to check which providers are configured:

```bash
python -c "
from ocr.providers import (
    EASYOCR_AVAILABLE, PADDLEOCR_AVAILABLE, DOCTR_AVAILABLE,
    TESSERACT_AVAILABLE, WINOCR_AVAILABLE,
    GEMINI_AVAILABLE, CLAUDE_AVAILABLE, OPENAI_VISION_AVAILABLE,
    AZURE_AVAILABLE, TEXTRACT_AVAILABLE,
    DEEPSEEK_AVAILABLE, FLORENCE_AVAILABLE, QWEN2VL_AVAILABLE
)

print('=== Local Providers ===')
print(f'  Tesseract: {TESSERACT_AVAILABLE}')
print(f'  EasyOCR: {EASYOCR_AVAILABLE}')
print(f'  PaddleOCR: {PADDLEOCR_AVAILABLE}')
print(f'  docTR: {DOCTR_AVAILABLE}')
print(f'  Windows OCR: {WINOCR_AVAILABLE}')

print()
print('=== Cloud Providers ===')
print(f'  Gemini Vision: {GEMINI_AVAILABLE}')
print(f'  Claude Vision: {CLAUDE_AVAILABLE}')
print(f'  OpenAI Vision: {OPENAI_VISION_AVAILABLE}')
print(f'  Azure Doc Intel: {AZURE_AVAILABLE}')
print(f'  AWS Textract: {TEXTRACT_AVAILABLE}')

print()
print('=== Local VLM (GPU Required) ===')
print(f'  DeepSeek OCR: {DEEPSEEK_AVAILABLE}')
print(f'  Florence-2: {FLORENCE_AVAILABLE}')
print(f'  Qwen2-VL: {QWEN2VL_AVAILABLE}')
"
```

---

## Cost Comparison (per 1,000 images)

| Provider | Cost | Notes |
|----------|------|-------|
| Local (Tesseract, etc.) | $0 | CPU only |
| Gemini Flash | ~$1 | Best value cloud |
| Claude Haiku | ~$0.40 | Cheapest LLM |
| GPT-4o-mini | ~$1 | Good balance |
| Google Vision | $1.50 | Traditional OCR |
| Azure Doc Intel | $1.50 | Enterprise |
| AWS Textract | $1.50 | AWS ecosystem |
| Claude Sonnet | ~$5 | Higher accuracy |
| GPT-4o | ~$8 | Highest accuracy |
