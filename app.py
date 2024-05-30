import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set the Tesseract path dynamically or from the environment variable
tesseract_path = os.getenv('TESSERACT_PATH')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Check common installation paths or set a default
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def biggestRectangle(contours):
    biggest = None
    max_area = 0
    indexReturn = -1
    for index in range(len(contours)):
        i = contours[index]
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.1 * peri, True)
            if area > max_area:
                biggest = approx
                max_area = area
                indexReturn = index
    return indexReturn

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Image could not be read'}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    invGamma = 1.0 / 0.3
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, table)

    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh1 = cv2.medianBlur(thresh1, 3)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    indexReturn = biggestRectangle(contours)
    if indexReturn == -1:
        return {'error': 'No valid contour found'}

    hull = cv2.convexHull(contours[indexReturn])

    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, indexReturn, 255, -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    (y, x, _) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img[topy:bottomy + 1, topx:bottomx + 1, :]

    config = '--psm 6 -c tessedit_char_whitelist=0123456789Vv'
    text = pytesseract.image_to_string(out, lang='eng', config=config)

    nic_numbers = re.findall(r'\b\d{9}[vV]?\b|\b\d{12}\b', text)
    
    return {'nic_numbers': nic_numbers}

import os

@app.route('/upload_image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        if filename.lower().endswith('.jifi'):
            try:
                image = Image.open(file_path)
                converted_path = file_path.rsplit('.', 1)[0] + '.png'
                image.save(converted_path)
                os.remove(file_path)
                file_path = converted_path
            except Exception as e:
                return jsonify({'error': f'Failed to convert .jifi file: {str(e)}'})
        
        result = process_image(file_path)
        os.remove(file_path)
        
        return jsonify(result)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
