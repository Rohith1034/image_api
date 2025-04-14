from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, BlipForConditionalGeneration

# Initialize the Flask app
app = Flask(__name__)

# Load the processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the caption generation function
def generate_caption(image):
    # Preprocess the image and text, then pass it to the model
    text = "A picture of"
    inputs = processor(images=image, text=text, return_tensors="pt")

    # Generate the caption
    outputs = model.generate(**inputs)

    # Decode the generated tokens to get the caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# Home route - Hello World
@app.route('/')
def home():
    return "Hello, World!"

# Define a route to handle image caption requests with image file upload
@app.route('/generate_caption', methods=['POST'])
def generate_caption_api():
    # Check if an image file is provided in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    # Check if the uploaded file is an image
    try:
        image = Image.open(image_file.stream)
    except Exception as e:
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 400
    
    # Generate the caption for the image
    caption = generate_caption(image)

    # Return the caption as JSON response
    return jsonify({'caption': caption})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
