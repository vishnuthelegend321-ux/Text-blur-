import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image
import io
import zipfile

# ==============================================================================
# App Title and Configuration
# ==============================================================================
st.set_page_config(layout="wide", page_title="Bulk Image Blurrer")

st.title("🚀 Bulk Image Text Blurrer")
st.write("Upload multiple images to blur text. Results are provided as a ZIP file.")

# ==============================================================================
# Important Warning for Large Batches
# ==============================================================================
st.warning(
    "**Disclaimer:** This free app has limited memory (RAM). "
    "Processing a very large number of files (e.g., 500-1000+) or very large images "
    "may cause the app to crash. It is best suited for smaller batches."
)

# ==============================================================================
# Cached OCR Model Loader
# ==============================================================================
@st.cache_resource
def get_ocr_reader():
    """Initializes and returns the EasyOCR reader, cached for performance."""
    return easyocr.Reader(['en'], gpu=False)

# ==============================================================================
# Core Image Processing Function
# ==============================================================================
def blur_text_in_image(image_bytes):
    """Detects and blurs text in an image and returns the processed image."""
    try:
        reader = get_ocr_reader()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = np.array(image)
        # Convert RGB (PIL) to BGR (OpenCV)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        detections = reader.readtext(image_cv)
        output_image = image_cv.copy()

        for (bbox, text, prob) in detections:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            roi = output_image[tl[1]:br[1], tl[0]:br[0]]
            # Use a strong blur for redaction
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
            output_image[tl[1]:br[1], tl[0]:br[0]] = blurred_roi

        # Convert back to RGB for display/saving
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        return output_image_rgb
    except Exception:
        # Return None on error to signify failure
        return None

# ==============================================================================
# Main App Logic with Progress Bar
# ==============================================================================
uploaded_files = st.file_uploader(
    "Upload your image files here",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"You have selected {len(uploaded_files)} images. Click the button to start.")

    if st.button(f"Start Processing {len(uploaded_files)} Images", type="primary"):
        # Create a zip file in memory to hold results
        zip_buffer = io.BytesIO()
        
        # --- NEW: Progress bar and status text ---
        progress_bar = st.progress(0, text="Starting...")
        status_text = st.empty()
        processed_count = 0
        
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress bar and status
                progress_text = f"Processing image {i + 1}/{total_files}: {uploaded_file.name}"
                progress_bar.progress((i + 1) / total_files, text=progress_text)
                status_text.text(progress_text)

                image_bytes = uploaded_file.getvalue()
                blurred_image = blur_text_in_image(image_bytes)
                
                if blurred_image is not None:
                    # Convert processed image back to bytes (PNG format)
                    pil_img = Image.fromarray(blurred_image)
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format="PNG")
                    
                    # Add the result to the zip file
                    zip_file.writestr(f"blurred_{uploaded_file.name}.png", img_buffer.getvalue())
                    processed_count += 1
                else:
                    status_text.warning(f"Skipped file due to an error: {uploaded_file.name}")

        progress_bar.empty() # Clear the progress bar
        status_text.success(f"✅ Processing complete! {processed_count}/{total_files} images were blurred successfully.")

        st.download_button(
            label="📁 Download All as ZIP",
            data=zip_buffer.getvalue(),
            file_name="blurred_images.zip",
            mime="application/zip",
        )
