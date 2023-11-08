
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image):
        self.image = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def equalize_histogram(self):
        self.image = cv2.equalizeHist(self.image)

    def auto_adjust_brightness(self):
        mean_val = np.mean(self.image)
        desired_brightness = 128
        brightness_adjust_value = desired_brightness - mean_val
        self.image = cv2.convertScaleAbs(self.image, alpha=1, beta=brightness_adjust_value)

    def auto_adjust_contrast(self):
        dst = np.zeros(self.image.shape, self.image.dtype)
        self.image = cv2.normalize(self.image, dst, 0, 255, cv2.NORM_MINMAX)

    def reduce_noise(self, kernel_size):
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

    def increase_resolution(self, scale_percent):
        width = int(self.image.shape[1] * scale_percent / 100)
        height = int(self.image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_LINEAR)

    # Adaptive Thresholding https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    def apply_adaptive_threshold(self, block_size, C):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

    # Bitwise Operation https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
    def invert_image(self):
        self.image = cv2.bitwise_not(self.image)

    def get_processed_image(self):
        return self.image

st.title("Fingerprint Image Processing")

uploaded_files = st.file_uploader("Upload fingerprint images", type=["bmp"], accept_multiple_files=True)

processed_images = []

if uploaded_files:
    for file in uploaded_files:
        # Convert BMP to PNG using PIL
        image = Image.open(file).convert("L")  # Convert to grayscale
        image_np = np.array(image)
        
        st.write(f"Settings for {file.name}")
        
        # Checkbox and Sliders
        use_hist_eq = st.checkbox("Use Histogram Equalization", value=False, key=f"hist_eq_{file.name}")
        use_noise_reduction = st.checkbox("Use Noise Reduction", value=True, key=f"noise_{file.name}")
        use_auto_brightness_contrast = st.checkbox("Use Auto Brightness and Contrast", value=True, key=f"auto_bright_cont_{file.name}")
        use_increase_resolution = st.checkbox("Increase Resolution", value=True, key=f"res_{file.name}")
        use_adaptive_threshold = st.checkbox("Use Adaptive Gaussian Threshold", value=True, key=f"adaptive_{file.name}")
        use_invert = st.checkbox("Invert Image after Thresholding", value=False, key=f"invert_{file.name}")
        show_histogram_option = st.checkbox("Show Histogram", value=False, key=f"hist_{file.name}")

        noise_size = 7
        if use_noise_reduction:
            noise_size = st.slider("Noise Reduction Kernel Size (odd value)", min_value=3, max_value=31, value=11, step=2, key=f"noise_kernel_{file.name}")

        resolution_scale = 200
        if use_increase_resolution:
            resolution_scale = st.slider("Resolution Scale Percentage", min_value=100, max_value=300, value=resolution_scale, step=10, key=f"res_scale_{file.name}")


        block_size = 21
        C = -7
        if use_adaptive_threshold:
            block_size = st.slider("Adaptive Gaussian Threshold Block Size (odd value)", 3, 31, block_size, step=2, key=f"block_{file.name}")
            C = st.slider("Adaptive Threshold C value", -15, 15, C, key=f"C_{file.name}")

        # Image processing
        processor = ImageProcessor(image_np)
        if use_hist_eq:
            processor.equalize_histogram()
        if use_auto_brightness_contrast:
            processor.auto_adjust_brightness()
            processor.auto_adjust_contrast()
        if use_noise_reduction:
            processor.reduce_noise(noise_size)
        if use_increase_resolution:
            processor.increase_resolution(resolution_scale)
        if use_adaptive_threshold:
            processor.apply_adaptive_threshold(block_size, C)
        if use_invert:
            processor.invert_image()

        binary_image = processor.get_processed_image()
        processed_images.append((f"binary_{file.name}.png", binary_image))
        
        # Display images
        col1, col2 = st.columns(2)
        col1.image(image, caption=f"Original Image - {file.name}", use_column_width=True)
        col2.image(binary_image, caption="Processed Image", use_column_width=True)

        # Histogram
        if show_histogram_option:
            hist = cv2.calcHist([binary_image], [0], None, [256], [0, 256])
            plt.figure(figsize=(10, 4))
            plt.plot(hist)
            plt.title(f'Histogram for {file.name}')
            plt.xlabel('Pixel value')
            plt.ylabel('Frequency')
            st.pyplot(plt)

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
