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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
from os.path import exists
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import util

from matplotlib import pyplot as plt

from BoundingBox import BoundingBox
from DetectedObject import DetectedObject
import image_helper
import cv2 as cv

from PIL import Image
import sys
include_path = '{}'.format(util.get_base_directory())
print(include_path)
sys.path.insert(0, include_path)
from include.models.research.object_detection.utils import label_map_util
from include.models.research.object_detection.utils import visualization_utils as vis_util

import constants
from constants import CameraType
from StereoMatch import StereoMatch

import time



def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with gfile.FastGFile(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  print(file_name)
  print(input_name)
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess1 = tf.Session()
  result = sess1.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


def label_image(file_name):
  # file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = '../model/output_graph.pb'
  label_file = '../model/output_labels.txt'
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = 'Placeholder'
  output_layer = 'final_result'

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess1:
    results = sess1.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  for i in top_k:
    print(labels[i], results[i])

  return_string = ""
  for label, result in zip(labels, results):
    return_string = return_string + str(label) + " "
    return_string = return_string + str(result) + " "

  return return_string


#
#
# INITIALIZE MODEL AND VARIABLES UPON IMPORT OF THIS SCRIPT
#
#
MINIMUM_CONFIDENCE = 0.98
MODEL_NAME = '{}/sign_model'.format(util.get_resources_directory())
PATH_TO_FROZEN_GRAPH = '{}/frozen_inference_graph.pb'.format(MODEL_NAME)
PATH_TO_LABELS = '{}/sign_labels/object-detection.pbtxt'.format(util.get_resources_directory())
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
sess = tf.Session(graph=detection_graph)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#
#
#
#
#

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image_np_expanded):
  with detection_graph.as_default():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    start_time = time.time()
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: image_np_expanded})
    print('tf session run time: {} seconds'.format(time.time() - start_time))

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict


def get_classification_dict_for_image(image_np):
  image_np_expanded = np.expand_dims(image_np, axis=0)
  output_dict = run_inference_for_single_image(image_np_expanded)

  # visualization of the results of a detection.
  # image_np_copy = np.copy(image_np)
  # vis_util.visualize_boxes_and_labels_on_image_array(
  #     image_np_copy,
  #     output_dict['detection_boxes'],
  #     output_dict['detection_classes'],
  #     output_dict['detection_scores'],
  #     category_index,
  #     instance_masks=output_dict.get('detection_masks'),
  #     use_normalized_coordinates=True,
  #     line_thickness=8)

  # show the label on the image
  # cv.imwrite('/tmp/image_label.jpg', image_np_copy)

  return output_dict


def get_string_for_classification_dict(d):
  result_string = ''

  for i, confidnece in enumerate(d['detection_scores']):
    if confidnece < MINIMUM_CONFIDENCE:
      break

    class_enum = constants.get_class_type_for_number(d['detection_classes'][i])
    result_string += '{} 15.2 35.2 '.format(str(class_enum))

  return result_string


def get_detection_list_for_classification_dict(classification_dict, image_width, image_height):
  boxes = classification_dict['detection_boxes']
  scores = classification_dict['detection_scores']
  classes = classification_dict['detection_classes']
  detection_list = []
  for i, box in enumerate(boxes):
    if scores[i] > MINIMUM_CONFIDENCE:
      bounding_box = BoundingBox(xmin=box[1]*image_width, xmax=box[3]*image_width, ymin=box[0]*image_height, ymax=box[2]*image_height)
      class_enum = constants.get_class_type_for_number(classes[i])
      print('Located {} at {}'.format(str(class_enum), str(bounding_box)))
      detection_list.append(DetectedObject(class_enum, bounding_box))

  return detection_list


def get_response_string_with_image_paths(image1_path, image2_path):
  camera_type_left = CameraType.PICAM_LEFT
  camera_type_right = CameraType.PICAM_RIGHT

  #
  # LOAD IMAGES FROM MEMORY
  #

  time_load_images = time.time()

  print("Undistorting Images")
  image1 = cv.imread(image1_path)
  # mtx1, dist1 = image_helper.get_calib_data_for_camera_type(camera_type_left)
  # image1 = image_helper.undistort(image1, mtx1, dist1)
  image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
  # TODO check to ensure classification works better for RGB images than BGR

  image2 = cv.imread(image2_path)
  # mtx2, dist2 = image_helper.get_calib_data_for_camera_type(camera_type_right)
  # image2 = image_helper.undistort(image2, mtx2, dist2)
  image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

  time_load_images = time.time() - time_load_images

  #
  # GET CLASSIFICATION DICT AND CONVERT TO OUR OWN CLASSIFICATION DICTIONARY OBJECTS
  #

  time_classification_dict = time.time()

  print("Classification step")
  image1_classification_dict = get_classification_dict_for_image(image1)
  detection_list = get_detection_list_for_classification_dict(image1_classification_dict, image1.shape[1], image1.shape[0])

  response_string = ''

  # ignore any computer vistion code if signs are not detection
  if len(detection_list) == 0:
    print('No detections found, returning empty string')
    return response_string

  # TODO handle case where homograohy matrix doesn't have a point in one or more of the bounding boxes (don't just crash out of nowhere)
  # TODO change all references to image1 to image_left and image2 as image_right

  time_classification_dict = time.time() - time_classification_dict

  #
  # GET HOMOGRAPHY MATRIX AND CONVERT MATCHES TO OUR MATCH OBJECT
  #

  time_homography_matrix = time.time()

  _, kpL, kpR, good, matchesMask = image_helper.get_homography_matrix(image1, image2)
  stereo_matches_list = []
  print("Calculating depth for each detection")
  for i, match in enumerate(matchesMask):
    if match:
      stereo_matches_list.append(StereoMatch(kpL[good[i].queryIdx].pt, kpR[good[i].trainIdx].pt))

  time_homography_matrix = time.time() - time_homography_matrix

  #
  # PARSE THROUGH MATCHES TO SEE IF WE CAN GET THE DEPTH FOR OUR OBJECTS
  #

  time_calculate_depth = time.time()

  for detection in detection_list:
    found_depth = False

    for match in stereo_matches_list:
      if util.is_in_box(match.left_pixel, detection.bounding_box):
        depth = image_helper.calculate_depth(match.left_pixel, match.right_pixel, camera_type_left)
        angle = image_helper.calculate_angle_to_pixel(image1, match.left_pixel, camera_type_left.get_horizontal_field_of_view())
        print((str(detection.class_type), depth, angle))
        response_string += '{} {} {} '.format(str(detection.class_type), depth, angle)
        found_depth = True
        break

    if not found_depth:
      # here is the case where there are no matches detected in the bounding box
      print('No feature matches located in the bounding box')
      angle = image_helper.calculate_angle_to_pixel(image1, detection.bounding_box.get_center_pixel(), camera_type_left.get_horizontal_field_of_view())
      response_string += '{} {} {} '.format(str(detection.class_type), constants.INVALID_MEASUREMENT, angle)

  time_calculate_depth = time.time() - time_calculate_depth

  #
  # FINISH AND RETURN RESPONSE
  #

  print('Label image step 1 load images time: {} seconds'.format(time_load_images))
  print('Label image step 2 classification dict time: {} seconds'.format(time_classification_dict))
  print('Label image step 3 homography matrix time: {} seconds'.format(time_homography_matrix))
  print('Label image step 4 calculate depth time: {} seconds'.format(time_calculate_depth))

  print("Finished, responding with response_string:" + response_string)
  response_string = response_string.rstrip() # remove tailing whitespace from response
  return response_string


def run():
  image_left_path = '/home/emersonboyd/git_repo/nu-eyegaze/eyegaze-scripts/src/image_left.jpg'
  image_right_path = '/home/emersonboyd/git_repo/nu-eyegaze/eyegaze-scripts/src/image_right.jpg'
  print(get_response_string_with_image_paths(image_left_path, image_right_path))
  print(get_response_string_with_image_paths(image_left_path, image_right_path))
  print(get_response_string_with_image_paths(image_left_path, image_right_path))


if __name__ == '__main__':
  run()
