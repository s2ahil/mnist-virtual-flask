# from flask import Flask
# import joblib
# import pickle
# import cv2
# import numpy as np
# from image_processing import preprocess

# from keras.models import model_from_json 

# # opening and store file in a variable

# json_file = open('myModel.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# loaded_model.load_weights("models_w.h5")

# image_path = '4.png'  # Replace with the path to your image


# # def preprocess():
# #  gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # # Resize the image to 28x28 pixels (MNIST size)
# #  resized_image = cv2.resize(gray_image, (28, 28))

# # # Invert the colors (black to white, white to black)
# #  inverted_image = cv2.bitwise_not(resized_image)


# #  return inverted_image


# app = Flask(__name__)



# @app.route('/')
# def hello_world():
#     # Read the uploaded image
# #  image_path = '4.png'  # Replace with the path to your image
#  pre_img = preprocess(image_path)


# #  def preprocess():
# #   gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # # Resize the image to 28x28 pixels (MNIST size)
# #   resized_image = cv2.resize(gray_image, (28, 28))

# # # Invert the colors (black to white, white to black)
# #   inverted_image = cv2.bitwise_not(resized_image)


# #   return inverted_image
 

#  # pre_img=preprocess()
#  prediction = loaded_model.predict(np.array([pre_img]))
#  predicted_digit = np.argmax(prediction)
#  print(predicted_digit)
 
#  return 'Hello, World BRO! hh  '


# if __name__ == '__main__':
#  app.run(host='0.0.0.0', port=5000, debug=True)


from fastapi import FastAPI, HTTPException, Form
from PIL import Image
import numpy as np
import base64
import io
import cv2
from keras.models import model_from_json

from fastapi.middleware.cors import CORSMiddleware



# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:5713",  # Add your React app's URL here
]

app = FastAPI()


json_file = open('myModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("models_w.h5")

image_path = '4.png'



@app.get("/")
async def index():
 def preprocess():
  gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels (MNIST size)
  resized_image = cv2.resize(gray_image, (28, 28))

# Invert the colors (black to white, white to black)
  inverted_image = cv2.bitwise_not(resized_image)


  return inverted_image
 

 pre_img=preprocess()
 prediction = loaded_model.predict(np.array([pre_img]))
 predicted_digit = np.argmax(prediction)
 print(predicted_digit)
 return {"message": "Hello World"+str(predicted_digit)}


@app.post("/process-canvas-image/")
async def process_canvas_image(dataURI: str = Form(...)):
    try:
        # Extract base64-encoded image data from dataURI
        encoded_data = dataURI.split(",")[1]
        decoded_image = base64.b64decode(encoded_data)

        # Convert image data to OpenCV format
        image_array = np.array(Image.open(io.BytesIO(decoded_image)))

        # Preprocess the image
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(gray_image, (28, 28))
        inverted_image = cv2.bitwise_not(resized_image)
        pre_img = inverted_image.reshape(1, 28, 28, 1).astype('float32') / 255

        # Predict digit using loaded model
        prediction = loaded_model.predict(pre_img)
        predicted_digit = np.argmax(prediction)

        return {"predicted_digit": int(predicted_digit)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))









