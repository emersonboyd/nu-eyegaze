import os


def file_exists(path):
	return os.path.exists(path)


def get_resources_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/../res'.format(this_directory)
	

def get_output_directory():
	this_directory = os.path.dirname(os.path.abspath(__file__))
	return '{}/../out'.format(this_directory)