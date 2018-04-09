#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import ops as utils_ops
from flask import Flask, request


if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = \
                        tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks,
                        detection_boxes,
                        image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = \
                tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def decode_prediction(output_dict, category_index):
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    min_score_thresh = .5
    if 0 == boxes.shape[0]:
        return '', 0

    if scores[0] > min_score_thresh:
        pic_label = category_index[classes[0]]['name']
        pic_prox = scores[0]
        return pic_label, pic_prox

    return '', 0


def return_encode(pic_label):

    if pic_label == 'fish_tofu':
        result_trans = {'name': u'鱼豆腐'}

    elif pic_label == 'roast_sausage':
        result_trans = {'name': u'烤肠'}

    elif pic_label == '':
        result_trans = {'name': u'未识别'}

    result_json = json.dumps(
        result_trans, encoding="UTF-8", ensure_ascii=False)
    return result_json


# Initialize flask application and the model
app = Flask(__name__)

PATH_TO_CKPT = '/root/volume/output/frozen_inference_graph.pb'
PATH_TO_LABELS = '/root/volume/data/label_map.pbtxt'
NUM_CLASSES = 2

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Aquire the upload file to be predict
        image = request.files['files']
        # basepath = os.path.dirname(__file__)
        # upload_path = os.path.join(basepath, 'demo.jpg')
        # image.save(upload_path)
        # image = Image.open(upload_path)
        image_np = load_image_into_numpy_array(image)

        # Actual object detection.
        output_dict = run_inference_for_single_image(
            image_np, detection_graph)

        # aquire the file label
        pic_label, pic_prox = decode_prediction(output_dict, category_index)
        print (pic_label, pic_prox)

        # encode result as format of JSON
        result_json = return_encode(pic_label)
        return result_json
    return 'Error Format'


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=6006)
