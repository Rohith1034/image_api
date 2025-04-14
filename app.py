from flask import Flask, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv  # <-- NEW
load_dotenv() 
import os

app = Flask(__name__)

# Read Hugging Face API token from environment variable
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
  # Don't hardcode your token here!

# Hugging Face model URL
API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/generate_caption", methods=["POST"])
def caption_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = Image.open(request.files['image'].stream).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes
        )

        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return jsonify({"caption": result[0]["generated_text"]})
        else:
            return jsonify({"error": "Failed to generate caption", "details": result}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
