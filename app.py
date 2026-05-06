import streamlit as st
import cv2
import numpy as np
from aging import age_effect

st.set_page_config(page_title="AI Face Aging", layout="wide")

st.title("👴 AI Face Aging Simulator")

st.write("Upload an image and simulate realistic aging effects")

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)

    age_level = st.slider(
        "Age Intensity",
        0.0,
        1.0,
        0.6
    )

    aged = age_effect(image, age_level)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original",
            channels="BGR"
        )

    with col2:
        st.image(
            aged,
            caption="Aged Output",
            channels="BGR"
        )

    success, buffer = cv2.imencode('.png', aged)

    st.download_button(
        label="Download Aged Image",
        data=buffer.tobytes(),
        file_name="aged_face.png",
        mime="image/png"
    )