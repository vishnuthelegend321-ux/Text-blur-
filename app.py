import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image
import io

# App Title and Description
st.set_page_config(layout="wide", page_title="Text Blurring App")
st.title("🖼️ Automatic Text Blurring App")
st.write("Upload an image, and this app will automatically detect and blur any text it finds.")

# Caching the OCR model to make it faster
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

# The function that blurs the text
def blur_text_in_image(image_bytes):
    try:
        reader = get_ocr_reader()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        detections = reader.readtext(image_cv)
        output_image = image_cv.copy()

        for (bbox, text, prob) in detections:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            roi = output_image[tl[1]:br[1], tl[0]:br[0]]
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
            output_image[tl[1]:br[1], tl[0]:br[0]] = blurred_roi

        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return output_image_rgb
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image_bytes, use_column_width=True)
    with col2:
        st.subheader("Blurred")
        with st.spinner("Processing..."):
            blurred_image = blur_text_in_image(image_bytes)
            if blurred_image is not None:
                st.image(blurred_image, use_column_width=True)
                pil_img = Image.fromarray(blurred_image)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="blurred_image.png",
                    mime="image/png"
          )
