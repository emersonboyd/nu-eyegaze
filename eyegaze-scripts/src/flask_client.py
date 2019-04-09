import numpy as np
import json
import requests
import random
import time

import util

NO_LABEL = "no_label"

def get_highest_confidence(labels_and_confidences):
    labels = []
    confidences = []
    labels_and_confidences = labels_and_confidences.text.rstrip().split(' ')
    print(len(labels_and_confidences))
    for i in range(0, len(labels_and_confidences), 2):
        print(i)
        labels.append(labels_and_confidences[i])
        confidences.append(labels_and_confidences[i+1])
    max_index = confidences.index(max(confidences))
    return labels[max_index]


def parse_labels(response_string):
    response_split = response_string.split(' ')
    label_list = []
    for i in range(0, len(response_split), 3):
        label_list.append(response_split[i])
    return label_list


def send_images(image_left_name, image_right_name, ip="73.100.56.111", port="5000", command="inf"):
    """
    Send an image to the Google Cloud Server

    """

    img1 = open(image_left_name, 'rb')
    print(image_left_name)
    files = {'image_left': img1}
    img2 = open(image_right_name, 'rb')
    print(image_right_name)
    files['image_right'] = img2
    post_command = "http://" + ip + ":" + port + "/" + command
    print(post_command)
    print(files)
    response = requests.post(post_command, files=files)
    return response


def main():
    command = 'hello'
    port = '5000'
    ip = '73.100.56.111'
    post_command = 'http://' + ip + ':' + port + '/' + command
    print(post_command)
    response = requests.post(post_command)
    print(response.text)


if __name__ == '__main__':
    main()
