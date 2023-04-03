import streamlit as st

st.set_page_config(
    page_title="Home"
)

st.title("CIFAR-10 Multi-Level Classification and Reverse Image Search")

st.markdown("**Welcome to our image classification and reverse search webapp! Our tool is powered by a convolutional neural network (CNN) trained on the CIFAR-10 dataset, which allows us to classify images and perform reverse image searches.  \nWith our webapp, you can upload an image and our model will first classify it into one of two categories, which are Living and Non-Living and then secondly into ten categories, including airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.  \nYou can also perform a reverse search by uploading an image and our model will find images similar to it in our database. Give it a try and discover what our model can do!**")

