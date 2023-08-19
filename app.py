from flask import Flask
import joblib
import cv2
import numpy as np
import keras_preprocessing
app = Flask(__name__)

model = joblib.load('mnist.joblib')

@app.route('/')
def hello_world():
    # Read the uploaded image
 image_path = '4.png'  # Replace with the path to your image



 def preprocess():
  gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels (MNIST size)
  resized_image = cv2.resize(gray_image, (28, 28))

# Invert the colors (black to white, white to black)
  inverted_image = cv2.bitwise_not(resized_image)


  return inverted_image
 

 pre_img=preprocess()
 prediction = model.predict(np.array([pre_img]))
 predicted_digit = np.argmax(prediction)

 print(predicted_digit)
 
 return 'Hello, World BRO! hh  d'+str(predicted_digit)


if __name__ == '__main__':
 app.run(debug=True)
