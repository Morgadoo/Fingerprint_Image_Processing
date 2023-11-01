
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

def process_image(image, brightness=0, contrast=1, use_noise_reduction=False, block_size=11, C=-2):
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    
    if use_noise_reduction:
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Adaptive Thresholding https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    
    # Bitwise Operation https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
    inverted_binary = cv2.bitwise_not(binary)
    
    return inverted_binary

st.title("Fingerprint Image Processing")

uploaded_file = st.file_uploader("Upload fingerprint images", type=["bmp"], accept_multiple_files=True)

processed_images = []

if uploaded_file:
    for file in uploaded_file:
        # Convert BMP to PNG using PIL
        image = Image.open(file).convert("L")  # Convert to grayscale
        image_np = np.array(image)
        
        st.write(f"Settings for {file.name}")
        
        # Checkbox and Sliders
        use_noise_reduction = st.checkbox("Use Noise Reduction", value=True, key=f"noise_{file.name}")
        
        brightness = st.slider("Brightness", -100, 100, 0, key=f"brightness_{file.name}")
        contrast = st.slider("Contrast", 0.5, 3.0, 1.0, key=f"contrast_{file.name}")

        block_size = st.slider("Adaptive Threshold Block Size (odd value)", 3, 31, 11, step=2, key=f"block_{file.name}")
        C = st.slider("Adaptive Threshold C value", -15, 5, -3, key=f"C_{file.name}")

        # Process the image
        binary_image = process_image(image_np, brightness, contrast, use_noise_reduction, block_size, C)     
        processed_images.append((f"binary_{file.name}.png", binary_image))
        
        # Create two columns display
        col1, col2 = st.columns(2)
        col1.image(image, caption=f"Original Image - {file.name}", use_column_width=True)
        col2.image(binary_image, caption="Binary Image", use_column_width=True, channels="GRAY")

    # ZIP all processed images
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
        for file_name, image in processed_images:
            img_buffer = io.BytesIO()
            img_pil = Image.fromarray(image)
            img_pil.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            zf.writestr(file_name, img_buffer.getvalue())
    zip_buffer.seek(0)

    st.download_button(label="Download All Processed Images", data=zip_buffer, file_name="processed_images.zip", mime="application/zip")
