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

r""" Fork form https://github.com/tensorflow/models/research/object_detection \
    dataset_tools/create_pascal_tf_record.py
Convert raw aggresss defined dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_tf_record.py \
        --data_dir=/home/user/xxxx
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_path', '',
                    'Root directory to aggresss defined dataset.')

FLAGS = flags.FLAGS


def check_dataset(dataset_directory):
    """Check the dataset directory is legal for aggresss dataset.

  Args:
    dataset_directory: Path to root directory holding aggresss dataset

  Returns:
    bool: ture or false
    dict: dict of breed

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid path
  """

    label_map_path = os.path.join(dataset_directory, 'label_map.pbtxt')
    if os.path.exists(label_map_path) is False:
        logging.warning("The dataset not contain the label map file.")
        return False, {}

    breed_dict = {}
    for item in os.listdir(dataset_directory):
        index_iamges = []
        index_annotations = []
        curr_path = os.path.join(dataset_directory, item)
        # Check is directory
        if os.path.isdir(curr_path):
            # Check 'iamges' and 'annotations' folders has the same elements
            images_path = os.path.join(curr_path, 'images')
            for i in os.listdir(images_path):
                filename, ext = os.path.splitext(i)
                if ext != '.jpg':
                    logging.warning("illegal file extension {}".format(i))
                    return False, {}
                index_iamges.append(filename)

            annotations_path = os.path.join(curr_path, 'annotations')
            for a in os.listdir(annotations_path):
                filename, ext = os.path.splitext(a)
                if ext != '.xml':
                    logging.warning("illegal file extension {}".format(a))
                    return False, {}
                index_annotations.append(filename)

            if index_annotations.sort() != index_iamges.sort():
                logging.warning(
                    "folder {} not has the same elements".format(item))
            else:
                breed_dict[item] = index_iamges

    if breed_dict != {}:
        print("found {} item in dataset directory".format(
            len(breed_dict)))
        return True, breed_dict

    return False, {}


def dict_to_tf_example(data,
                       image_path,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    image_path: Path to the image
    label_map_dict: A map from string label names to integers ids.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                dataset_util.int64_feature(height),
                'image/width':
                dataset_util.int64_feature(width),
                'image/filename':
                dataset_util.bytes_feature(data['filename'].encode('utf8')),
                'image/source_id':
                dataset_util.bytes_feature(data['filename'].encode('utf8')),
                'image/key/sha256':
                dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded':
                dataset_util.bytes_feature(encoded_jpg),
                'image/format':
                dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymax),
                'image/object/class/text':
                dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label':
                dataset_util.int64_list_feature(classes),
                'image/object/difficult':
                dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated':
                dataset_util.int64_list_feature(truncated),
                'image/object/view':
                dataset_util.bytes_list_feature(poses),
            }))
    return example


def main(_):
    data_path = FLAGS.data_path
    label_map_path = os.path.join(data_path, 'label_map.pbtxt')
    is_dataset, species = check_dataset(data_path)
    if is_dataset is True:
        print(species)

        writer_train = tf.python_io.TFRecordWriter('train.record')
        writer_val = tf.python_io.TFRecordWriter('val.record')

        label_map_dict = label_map_util.get_label_map_dict(label_map_path)

        for specie in species:
            logging.info('Reading from %s dataset.', specie)
            examples_list = species[specie]

            random.seed(42)
            random.shuffle(examples_list)
            # Set 70% data to train, 30% data to validation
            num_train = int(0.7 * len(examples_list))

            for idx, example in enumerate(examples_list):

                img_path = os.path.join(data_path, specie,
                                        'images', example + '.jpg')
                ann_path = os.path.join(data_path, specie,
                                        'annotations', example + '.xml')

                with tf.gfile.GFile(ann_path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = \
                    dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_example = dict_to_tf_example(data, img_path, label_map_dict)

                if idx < num_train:
                    writer_train.write(tf_example.SerializeToString())
                else:
                    writer_val.write(tf_example.SerializeToString())

        writer_train.close()
        writer_val.close()


if __name__ == '__main__':
    tf.app.run()
