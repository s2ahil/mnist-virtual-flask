from flask import Flask
import joblib
import pickle
import cv2
import numpy as np

from keras.models import model_from_json 

# opening and store file in a variable

json_file = open('myModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("models_w.h5")

image_path = '4.png'  # Replace with the path to your image


def preprocess():
 gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels (MNIST size)
 resized_image = cv2.resize(gray_image, (28, 28))

# Invert the colors (black to white, white to black)
 inverted_image = cv2.bitwise_not(resized_image)


 return inverted_image


app = Flask(__name__)



@app.route('/')
def hello_world():
    # Read the uploaded image
#  image_path = '4.png'  # Replace with the path to your image



#  def preprocess():
#   gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Resize the image to 28x28 pixels (MNIST size)
#   resized_image = cv2.resize(gray_image, (28, 28))

# # Invert the colors (black to white, white to black)
#   inverted_image = cv2.bitwise_not(resized_image)


#   return inverted_image
 

 pre_img=preprocess()
 prediction = model.predict(np.array([pre_img]))
 predicted_digit = np.argmax(prediction)
 print(predicted_digit)
 
 return 'Hello, World BRO! hh  d'+str(predicted_digit)


if __name__ == '__main__':
 app.run(debug=True)
