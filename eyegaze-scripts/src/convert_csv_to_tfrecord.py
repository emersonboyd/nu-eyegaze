# Adapted from
# https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from include.models.research.object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

import util


# Add more class labels as needed, make sure to start at 1
def class_text_to_int(row_label):
    if row_label == 'exit sign':
        return 1
    if row_label == 'bathroom sign':
        return 2
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    photo_path = '{}/sign_photos'.format(util.get_resources_directory())
    label_path = '{}/sign_labels'.format(util.get_resources_directory())
    for dataset_type in ['train', 'test']:
        photo_path_current = '{}/{}'.format(photo_path, dataset_type)
        label_path_current = '{}/{}.csv'.format(label_path, dataset_type)
        writer = tf.python_io.TFRecordWriter('{}/{}.record'.format(label_path, dataset_type))
        examples = pd.read_csv(label_path_current)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, photo_path_current)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the {} TFRecords'.format(label_path))


if __name__ == '__main__':
    tf.app.run()
    # main()
