import requests
import base64
import numpy as np
import cv2
from PIL import Image

url = 'http://localhost:8000/get_detection_data/'
response = requests.get(url)
data = response.json()
encoded_image = data['frame']

# Decodificar la imagen base64 en una matriz NumPy
image = np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8)
print(data['people_count'])

image_out = Image.frombytes("RGB", (640, 480),bytes(image))
image_out.save('image.png')

