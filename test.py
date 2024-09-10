import requests
import base64

# Read the image file and encode it to base64
with open("image1.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predicting/"

# Create the payload
payload = {
    "image": encoded_string
}

# Send the POST request
response = requests.post(url, json=payload)
# response = requests.get(url)

# Print the response
print(response.json())
