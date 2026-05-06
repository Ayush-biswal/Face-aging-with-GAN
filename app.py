import streamlit as st
import cv2
import numpy as np
from aging import age_effect

st.title("👴 AI Face Aging Simulator")

st.write("Upload an image to see aging effect 👇")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Original", channels="BGR")

    age_level = st.slider("Select Age Intensity", 0.0, 1.0, 0.5)

    aged = age_effect(img, age_level)

    st.image(aged, caption="Aged Output", channels="BGR")
else:
    st.warning("Please upload an image to continue")

# import streamlit as st

# st.title("WORKING TEST")
# st.write("If you see this, Streamlit is fine")