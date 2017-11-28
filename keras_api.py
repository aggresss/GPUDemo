#!/usr/bin/python
# Use flask build a API for Keras predict image
# curl -m 300 -F "file=@demo.jpg;type=image/jpeg" http://127.0.0.1:6006/upload
#

from flask import Flask,request,redirect,url_for
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os, json


app = Flask(__name__)
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'demo.jpg')
        f.save(upload_path)

        img = image.load_img(upload_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        model = VGG16(weights='imagenet')
        features = model.predict(x)
        result_list = list(decode_predictions(features, top=1)[0][0])
        result_trans = [ str(x) for x in result_list ]
        result_json = json.dumps(result_trans)
        return result_json + '\n'
    return 'Error Format'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6006)

