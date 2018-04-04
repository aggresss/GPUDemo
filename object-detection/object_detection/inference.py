# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model inference function for object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import glob
import argparse
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", help="Path of the input images directory")
parser.add_argument("--frozen_graph", help="Path of the frozen graph model")
parser.add_argument("--label_map", help="Path of the label map file")
parser.add_argument("--output_dir", help="Path of the output directory")
parser.add_argument("--num_output_classes",
                    help="Defines the number of output classes", type=int)

args = parser.parse_args()
PATH_TO_CKPT = args.frozen_graph
PATH_TO_LABELS = args.label_map
NUM_CLASSES = args.num_output_classes
PATH_TO_TEST_IMAGES_DIR = args.input_dir
PATH_TO_RESULT_IMAGES_DIR = args.output_dir


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def main(_):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    TEST_IMAGE_PATHS = glob.glob(
        os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

    JPG_PATHS = [os.path.basename(path) for path in TEST_IMAGE_PATHS]

    RESULT_IMAGE_PATHS = [os.path.join(
        PATH_TO_RESULT_IMAGES_DIR, jpg_path) for jpg_path in JPG_PATHS]

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image
            # where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')

            # Each score represent how level of confidence for
            # each of the objects. Score is shown on the result image,
            # together with the class label.

            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            count = 0

            for image_path, result_path in \
                    zip(TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS):

                image_np = Image.open(image_path)

                # the array based representation of the image will be used
                # later in order to prepare the
                # result image with boxes and labels on it.

                # Expand dimensions since the model expects images to have
                # shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                image_np.save(result_path)
                count += 1
                print('Images Processed:', count, end='\r')


if __name__ == '__main__':
    tf.app.run()
