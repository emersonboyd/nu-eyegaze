import os


def file_exists(path):
	return os.path.exists(path)


def get_base_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/..'.format(this_directory)


def get_resources_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/../res'.format(this_directory)
	

def get_output_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/../out'.format(this_directory)


def get_object_detection_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/../include/models/research/object_detection'.format(this_directory)


def pixel_in_bounds(image, pixel):
	x, y = pixel
	image_height, image_width = image.shape

	if x < 0 or y < 0:
		return False

	if x >= image_width:
		return False

	if y >= image_height:
		return False

	return True
