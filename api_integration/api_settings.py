"""
API endpoint URLs for Orah Vision AWS services
"""

class API_URLS:
    BASE_URL = "https://69kgnscoyk.execute-api.us-east-2.amazonaws.com/prod"
    TOKENS = f"{BASE_URL}/tokens"
    REGISTER = f"{BASE_URL}/register"
    TTS = "https://u3oluwnaq5v4yql2ixvhbthjde0bqkot.lambda-url.us-east-2.on.aws/"
    TTS_STREAMING = "https://3pydsw25r645mjzauwkxuwgsua0diaox.lambda-url.us-east-2.on.aws/tts-stream"
