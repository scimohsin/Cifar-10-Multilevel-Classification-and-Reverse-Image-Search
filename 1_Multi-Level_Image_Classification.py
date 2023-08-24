import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import cv2

def label_living(num):
    if num == 0:
        return 'Bird'
    elif num == 1:
        return 'Cat'
    elif num == 2:
        return 'Deer'
    elif num == 3:
        return 'Dog'
    elif num == 4:
        return 'Frog'
    elif num == 5:
        return 'Horse'

def label_nonliving(num):
    if num == 0:
        return 'Airplane'
    elif num == 1:
        return 'Automobile'
    elif num == 2:
        return 'Ship'
    elif num == 3:
        return 'Truck'

def label_binary(num):
    if num == 0:
        return 'Living'
    elif num == 1:
        return 'Non-Living'

binary = tf.keras.models.load_model('/content/pages/binary.h5')
living = tf.keras.models.load_model('/content/pages/living.h5')
nonliving = tf.keras.models.load_model('//content/pages/nonliving.h5')

def import_predict(image_data, model):
  size = (32, 32)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img_reshape = img[np.newaxis, ...]
  prediction = model.predict(img_reshape)

  return prediction

st.title("10 Class Multi-Level Classification")

file = st.file_uploader("Choose Image For Classification", type=["jpg", "png"])

if file is None:
  st.text("Please Upload An Image File!")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  prediction_first = import_predict(image, binary)
  string_first="This image most likely is: "+label_binary(np.argmax(prediction_first))
  if np.argmax(prediction_first) == 0:
    prediction_second = import_predict(image, living)
    string_second="This image most likely is: "+label_living(np.argmax(prediction_second))
  elif np.argmax(prediction_first) == 1:
    prediction_second = import_predict(image, nonliving)
    string_second="This image most likely is: "+label_nonliving(np.argmax(prediction_second))
  st.success(string_first)
  st.success(string_second)
  

