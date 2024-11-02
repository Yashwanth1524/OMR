import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI()

# Define paths
img_name = 'cleaned_sheet'
imgs_path = '/Users/yashwanthr/Downloads'
cleaned_images_path = os.path.join(imgs_path, f'cleaned_images_{img_name}')
os.makedirs(cleaned_images_path, exist_ok=True)

# HTML for the UI
html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>Upload Noisy Musical Sheet</title>
    </head>
    <body>
        <h1>Optical Music Recognition: Denoising Tool</h1>
        <p>Please upload a noisy musical sheet image to process and clean it.</p>
        <form action="/upload_and_denoise/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*">
            <input type="submit" value="Upload and Denoise">
        </form>
    </body>
</html>
"""

# Denoising function
def denoise_image(image):
    # Perform denoising
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=20, templateWindowSize=17, searchWindowSize=28)

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Create an output image to hold the components
    output_image = np.zeros_like(binary_image)

    # Iterate through each component
    for i in range(1, num_labels):
        # Create a mask for the current component
        component_mask = (labels == i).astype("uint8") * 255
        # Keep components larger than a certain threshold size to preserve more details
        if stats[i, cv2.CC_STAT_AREA] > 50:
            output_image = cv2.bitwise_or(output_image, component_mask)

    # Invert the image back to original white objects
    output_image = cv2.bitwise_not(output_image)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(output_image, -1, kernel)

    # Adjust contrast and brightness
    alpha = 1.5  # Contrast control
    beta = 50    # Brightness control
    enhanced_image = cv2.convertScaleAbs(sharpened_image, alpha=alpha, beta=beta)

    # Automatic border detection and cropping
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop based on the largest contour found
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = enhanced_image[y:y+h, x:x+w]
    else:
        cropped_img = enhanced_image  # In case no contours are found

    return cropped_img

# FastAPI route for the root with UI form
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

# FastAPI route to upload image, process, and denoise
@app.post("/upload_and_denoise/")
async def upload_and_denoise(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    cleaned_image = denoise_image(image)  # Denoise the uploaded image

    # Save cleaned image
    cleaned_image_path = os.path.join(cleaned_images_path, 'cleaned_image.png')
    cv2.imwrite(cleaned_image_path, cleaned_image)

    # Display the cleaned image in the response
    plt.figure(figsize=(10, 10))
    plt.imshow(cleaned_image, cmap='gray')
    plt.title('Final Cleaned Image')
    plt.axis('off')
    plt.show()

    return HTMLResponse(f"<h3>Image processed and cleaned!</h3><p>Check the console for the cleaned image display.</p>")

# Static files for serving images (if needed)
app.mount("/static", StaticFiles(directory=cleaned_images_path), name="static")

# Run the app using:
# uvicorn your_fastapi_script:app --reload
