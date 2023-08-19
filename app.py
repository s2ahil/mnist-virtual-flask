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


from fastapi import FastAPI

import joblib
import pickle
import cv2
import numpy as np
from image_processing import preprocess
from keras.models import model_from_json 



app = FastAPI()


json_file = open('myModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("models_w.h5")

image_path = '4.png



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












