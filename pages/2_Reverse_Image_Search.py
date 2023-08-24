import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import cv2
from keras.utils import to_categorical
#from sklearn.neighbors import KDTree
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1 = unpickle('data_batch_1')
batch_2 = unpickle('data_batch_2')
batch_3 = unpickle('data_batch_3')
batch_4 = unpickle('data_batch_4')
batch_5 = unpickle('data_batch_5')
test_batch = unpickle('test_batch')

batch1_data = batch_1[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
batch2_data = batch_2[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
batch3_data = batch_3[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
batch4_data = batch_4[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
batch5_data = batch_5[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
batch6_data = test_batch[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

batch1_labels = batch_1[b'labels']
batch2_labels = batch_2[b'labels']
batch3_labels = batch_3[b'labels']
batch4_labels = batch_4[b'labels']
batch5_labels = batch_5[b'labels']
batch6_labels = test_batch[b'labels']

train_data = np.concatenate((batch1_data, batch2_data, batch3_data, batch4_data, batch5_data, batch6_data), axis=0)
train_labels = np.concatenate((batch1_labels, batch2_labels, batch3_labels, batch4_labels, batch5_labels, batch6_labels), axis=0)

train_data = train_data.astype('float32') / 255
train_labels = to_categorical(train_labels)


def label(num):
    if num == 0:
        return 'Airplane'
    elif num == 1:
        return 'Automobile'
    elif num == 2:
        return 'Bird'
    elif num == 3:
        return 'Cat'
    elif num == 4:
        return 'Deer'
    elif num == 5:
        return 'Dog'
    elif num == 6:
        return 'Frog'
    elif num == 7:
        return 'Horse'
    elif num == 8:
        return 'Ship'
    elif num == 9:
        return 'Truck'

reverse = tf.keras.models.load_model('/content/pages/reverse.h5')
features = pickle.load(open('/content/pages/features.pkl','rb'))
tree = pickle.load(open('/content/pages/tree.pkl','rb'))

def import_search(image_data, model):
  size = (32, 32)
  image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img = np.asarray(image)
  img = np.expand_dims(img, axis=0)
  img = img.astype('float32') / 255
  img_features = model.predict(img)
  dist, ind = tree.query(img_features.reshape(img_features.shape[0], -1), k=10)
  return dist, ind

st.title("Reverse Image Search")
file = st.file_uploader("Choose Image For Searching", type=["jpg", "png"])

if file is None:
  st.text("Please Upload An Image File!")
else:
  image = Image.open(file)
  st.image(image, use_column_width=True)
  dist, ind = import_search(image, reverse)
  sim_img = []
  sim_label = []
  for i in range(ind.shape[1]):
    sim_img.append(train_data[ind[0,i],:,:,:])
    sim_label.append(np.argmax(train_labels[ind[0,i]]))

  col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)

  with col1:
      st.image(sim_img[0])
      st.write(label(sim_label[0]))
  with col2:
      st.image(sim_img[1])
      st.write(label(sim_label[1]))
  with col3:
      st.image(sim_img[2])
      st.write(label(sim_label[2]))
  with col4:
      st.image(sim_img[3])
      st.write(label(sim_label[3]))
  with col5:
      st.image(sim_img[4])
      st.write(label(sim_label[4]))
  with col6:
      st.image(sim_img[5])
      st.write(label(sim_label[5]))
  with col7:
      st.image(sim_img[6])
      st.write(label(sim_label[6]))
  with col8:
      st.image(sim_img[7])
      st.write(label(sim_label[7]))
  with col9:
      st.image(sim_img[8])
      st.write(label(sim_label[8]))
  with col10:
      st.image(sim_img[9])
      st.write(label(sim_label[9]))
