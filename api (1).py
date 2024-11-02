import os
import subprocess
import cv2
import re
import numpy as np
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Define paths
img_name = 'love'
imgs_path = '/Users/yashwanthr/Downloads'
save_folder = os.path.join(imgs_path, f'connected_components_{img_name}')
os.makedirs(save_folder, exist_ok=True)
test_folder = f"/Users/yashwanthr/Downloads/segmented_images_{img_name}"
ctc_predict_path = '/Users/yashwanthr/Downloads/tf-end-to-end-master/ctc_predict.py'
abc_notation_path = '/Users/yashwanthr/Downloads/abc_notation.txt'

# HTML for the UI
html_content = """
<!DOCTYPE html>
<html>
    <head>
        <title>Upload Musical Sheet</title>
    </head>
    <body>
        <h1>Welcome to the Optical Music Recognition API!</h1>
        <p>Please upload a musical sheet image to process and get ABC notation.</p>
        <form action="/upload_and_predict/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*">
            <input type="submit" value="Upload and Predict">
        </form>
    </body>
</html>
"""

# Preprocessing function
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return morphed

# Extract connected components
def extract_connected_components(image):
    labeled_img = label(image, connectivity=2)
    components = []
    for region in regionprops(labeled_img):
        min_row, min_col, max_row, max_col = region.bbox
        component = image[min_row:max_row, min_col:max_col]
        components.append(component)
    return components

# Save segmented components
def save_components(imgs_with_staff):
    counter = 1
    for i, img_with_staff in enumerate(imgs_with_staff):
        if img_with_staff is not None:
            img_with_staff_inverted = cv2.bitwise_not(img_with_staff)
            processed_image = preprocess_image(img_with_staff_inverted)
            connected_components = extract_connected_components(processed_image)
            for j, component in enumerate(connected_components):
                component_inverted = cv2.bitwise_not(component)
                file_name = f"{counter}.jpg"
                file_path = os.path.join(save_folder, file_name)
                cv2.imwrite(file_path, component_inverted)
                counter += 1

# Prediction and ABC notation conversion
def predict_and_output():
    abc_dict = {}
    with open(abc_notation_path, 'r') as abc_file:
        for line in abc_file:
            if ':' in line:
                text_representation, abc_format = line.strip().split(':', 1)
                abc_dict[text_representation.strip()] = abc_format.strip()

    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()) if re.search(r'(\d+)', x) else float('inf'))

    max_files_to_parse = 150
    image_files = image_files[:max_files_to_parse]
    valid_image_paths = []
    predictions = []

    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        model_path = '/Users/yashwanthr/Downloads/Semantic-Model 2/semantic_model.meta'
        vocabulary_path = '/Users/yashwanthr/Downloads/tf-end-to-end-master/Data/vocabulary_semantic.txt'
        command = ['python3', ctc_predict_path, '-image', image_path, '-model', model_path, '-vocabulary', vocabulary_path]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            prediction = result.stdout.strip()
            prediction_elements = [elem.strip() for elem in prediction.split('\n') if elem.strip()]
            if len(prediction_elements) >= 3:
                valid_image_paths.append(image_path)
                predictions.append(prediction)
                abc_formats = [abc_dict.get(elem, 'Unknown') for elem in prediction_elements]
                print(f'Processed {image_file}, Prediction: {prediction}')
                print('ABC Format:')
                for abc in abc_formats:
                    print(abc)

        except subprocess.CalledProcessError as e:
            print(f'Error processing {image_file}: {e.stderr}')

    # Reconstruct the musical sheet
    images = []
    target_width = None
    for image_path in valid_image_paths:
        image = cv2.imread(image_path)
        if target_width is None:
            target_width = image.shape[1]
        if image.shape[1] != target_width:
            new_height = int(image.shape[0] * (target_width / image.shape[1]))
            image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)
        images.append(image)

    if images:
        reconstructed_sheet = np.vstack(images)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(reconstructed_sheet, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No valid segments to reconstruct the musical sheet.")

# FastAPI route for the root with UI form
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

# FastAPI route to upload image, process, and predict
@app.post("/upload_and_predict/")
async def upload_and_predict(file: UploadFile = File(...)):
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    save_components([image])  # Save connected components

    # After saving the components, predict and display output
    predict_and_output()

    # Assuming the result is displayed via plt.show(), no file needs to be returned
    return HTMLResponse("<h3>Image processed and ABC notation predicted!</h3><p>Check the console for output.</p>")

# Static files for serving images (if needed)
app.mount("/static", StaticFiles(directory=save_folder), name="static")
