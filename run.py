from flask import Flask, render_template, request, jsonify, url_for
import cv2
from openai import AzureOpenAI
import base64
import numpy as np
import time
import os

api_base = 'https://AZURE_OPENAI_ENDPOINT.openai.azure.com/'
api_key= 'AZURE_OPENAI_API_KEY'
deployment_name = 'AZURE_OPENAI_API_GPT4O_DEPLOYMENT_NAME'
api_version = '2024-02-01'
# api_version reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference

# Initialize the Flask app
app = Flask(__name__)

# Define the home route and the function to render the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the process route and the function to process the image
@app.route('/process', methods=['POST'])
def process_image():
    data = request.data.decode('utf-8')  # Decode the bytes to a string
    encoded_data = data.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Define the upload folder
    upload_folder = "static/uploads"

    # Generate a unique filename
    filename = f"{time.time()}.jpg"
    filepath = os.path.join(upload_folder, filename)

    # Save the image
    cv2.imwrite(filepath, img)

    # Generate the image URL to be returned and the file path
    image_url = url_for('static', filename=f'uploads/{filename}', _external=True)
    image_url = filepath

    return jsonify(image=image_url)

# Define the chat route and the function to chat with the GPT-4o model
@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    image_url = request.form.get('image_url')

    print(image_url)

    # Open the image in binary mode
    with open(image_url, 'rb') as image_file:
        image_data = image_file.read()

    # Encode the image data to a base64 string
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    print(encoded_image)

    # Pass the message and the image to the GPT-4 model
    client = AzureOpenAI(
        api_key=api_key,  
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )

    # Define the user content
    user_content = [{"type": "text", "text": message}]
    if image_url:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}})

    print(user_content)

    # Get the response from the GPT-4o model
    response = client.chat.completions.create(
        model = deployment_name,
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': user_content}
        ]
    )

    # Convert the response to a dictionary
    response_dict = response.to_dict()
    # Return the response
    return jsonify(response=response_dict['choices'][0]['message']['content'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)