from __future__ import division, print_function
import os
import numpy as np
import cv2
from keras.models import load_model
import time

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

mod = load_model('model/ddd.hdf5')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)
        test_arr = []
        for i in range(1):
            img = cv2.imread(file_path, 0)
            resized = cv2.resize(img, (32, 32))
            test_arr.append(resized)
        test_arr = np.array(test_arr)
        test_arr_norm = test_arr / 255
        start = time.time()
        test_output_arr = mod.predict(test_arr_norm)
        end = time.time()
        print(start, end)
        print(end-start)
        #test_output_arr
        #print(test_output_arr)
        if np.argmax(test_output_arr) == 1:
            return "Distracted Driving --> Texting - right (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 2:
            return "Distracted Driving --> Talking on the phone - right (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 3:
            return "Distracted Driving --> Texting - left (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 4:
            return "Distracted Driving --> Talking on the phone - left (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 5:
            return "Distracted Driving --> Operating the radio (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 6:
            return "Distracted Driving --> Drinking (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 7:
            return "Distracted Driving --> Reaching behind (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 8:
            return "Distracted Driving --> Hair and makeup (Prediction time: {:.2f} secs)".format(end-start)
        elif np.argmax(test_output_arr) == 9:
            return "Distracted Driving --> Talking to passenger (Prediction time: {:.2f} secs)".format(end-start)
        else:
            return "Safe Driving (Prediction time: {:.2f} secs)".format(end-start)
    return None


if __name__ == '__main__':
    app.run(debug=True)
