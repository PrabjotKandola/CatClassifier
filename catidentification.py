import tensorflow as tf
from tensorflow import keras
import numpy as np
import urllib.request
from PIL import Image
import requests
from io import BytesIO

model = keras.applications.MobileNetV2(weights="imagenet")

def classify_cat(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

    return decoded_predictions[0]

image_url = "URL_TO_YOUR_CAT_IMAGE"
result = classify_cat(image_url)
print(f"Predicted cat breed: {result[1]}, Probability: {result[2]:.2%}")
