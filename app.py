from flask import Flask, request, render_template, send_from_directory, url_for
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

# Helper functions for each image processing operation
def apply_kmeans_compression(image_data, compression_level):
    k_values = {'low': 8, 'medium': 16, 'high': 32}
    k = k_values[compression_level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    compressed_img_data = kmeans.cluster_centers_[kmeans.labels_]
    return compressed_img_data.reshape(original_shape).astype(np.uint8)

def apply_noise_reduction(image_data, reduction_type, level):
    ksize_values = {'low': 3, 'medium': 5, 'high': 7}
    ksize = ksize_values[level]
    if reduction_type == 'median':
        return cv2.medianBlur(image_data, ksize)
    elif reduction_type == 'gaussian':
        return cv2.GaussianBlur(image_data, (ksize, ksize), 0)
    return image_data

def apply_image_sharpening(image_data, level):
    kernel_strength = {'low': -1, 'medium': -3, 'high': -5}
    kernel = np.array([[0, kernel_strength[level], 0], [kernel_strength[level], 9, kernel_strength[level]], [0, kernel_strength[level], 0]])
    return cv2.filter2D(image_data, -1, kernel)

def convert_to_grayscale(image_data, level):
    return cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

def apply_segmentation(image_data, level):
    k_values = {'low': 2, 'medium': 4, 'high': 8}
    k = k_values[level]
    original_shape = image_data.shape
    img_data_flattened = image_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_data_flattened)
    segmented_img_data = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img_data.reshape(original_shape).astype(np.uint8)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        operation = request.form.get('operation')
        level = request.form.get('level', 'low')
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = Image.open(filepath)
        img_data = np.array(img)

        processed_img_data = None
        if operation == 'compress':
            processed_img_data = apply_kmeans_compression(img_data, level)
        elif operation == 'reduce_noise':
            noise_type = request.form.get('noise_type', 'median')
            processed_img_data = apply_noise_reduction(img_data, noise_type, level)
        elif operation == 'sharpen':
            processed_img_data = apply_image_sharpening(img_data, level)
        elif operation == 'grayscale':
            processed_img_data = convert_to_grayscale(img_data, level)
        elif operation == 'segment':
            processed_img_data = apply_segmentation(img_data, level)

        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        Image.fromarray(processed_img_data).save(output_path)

        return render_template('result.html', image_url=url_for('get_file', filename=output_filename))

    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
