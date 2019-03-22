import os
import constants


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
	image_height, image_width, _ = image.shape

	if x < 0 or y < 0:
		return False

	if x >= image_width:
		return False

	if y >= image_height:
		return False

	return True


def is_in_box(pixel, box):
	x = pixel[0]
	y = pixel[1]

	return x >= box.xmin and x <= box.xmax and y >= box.ymin and y <= box.ymax


def parse_server_response(s):
	response_list = []
	print("Length of string: " + str(len(s)))
	s_split = s.split(' ')

	if len(s_split) == 1:
		return response_list

	for i in range(0, len(s_split), 3):
		print("Length of string split: " + str(len(s_split)))
		class_audio_file = get_class_audio_file(s_split[i])
		distance_audio_file = get_distance_audio_file(s_split[i+1])
		angle_audio_file = get_angle_audio_file(s_split[i+2])
		response_list.append((class_audio_file, distance_audio_file, angle_audio_file))

	return response_list


def get_class_audio_file(class_string):
	print("class_string: " + class_string)
	print("class_string enum: " + str(constants.get_class_type_for_string(class_string)))
	print("exit_string raw enum: " + str(constants.get_class_type_for_string('exit_sign')))
	return constants.get_class_type_for_string(class_string).get_audio_file_name()


def get_distance_audio_file(distance_float_m):
	if distance_float_m == constants.INVALID_MEASUREMENT:
		return 'feet_invalid.mp3'
	distance_float_m = float(distance_float_m)

	# invalid distance measurement
	if distance_float_m < 0:
		return 'feet_invalid.mp3'

	# convert distance in meters to distance in feet
	distance_float_ft = distance_float_m * 3.28084

	# return the correct mp3 filename
	if distance_float_ft >= 0 and distance_float_ft < 7.5:
		return 'feet_5.mp3'
	elif distance_float_ft < 12.5:
                return 'feet_10.mp3'
	elif distance_float_ft < 17.5:
                return 'feet_15.mp3'
	elif distance_float_ft < 22.5:
                return 'feet_20.mp3'
	elif distance_float_ft < 27.5:
                return 'feet_25.mp3'
	elif distance_float_ft < 32.5:
                return 'feet_30.mp3'
	else:
		return 'feet_greater_than_30.mp3'


def get_angle_audio_file(angle_float):
        if angle_float == constants.INVALID_MEASUREMENT:
                return 'angle_invalid.mp3'
        angle_float = float(angle_float)

        # invalid distance measurement
        if angle_float < -55 or angle_float >= 55:
                return 'angle_invalid.mp3'

        # return the correct mp3 filename
        if angle_float >= -55 and angle_float < -45:
                return 'left_50.mp3'
        elif angle_float >= -45 and angle_float < -35:
                return 'left_40.mp3'
        elif angle_float >= -35 and angle_float < -25:
                return 'left_30.mp3'
        elif angle_float >= -25 and angle_float < -15:
                return 'left_20.mp3'
        elif angle_float >= -15 and angle_float < -5:
                return 'left_10.mp3'
        elif angle_float >= -5 and angle_float < 5:
                return 'straight.mp3'
        elif angle_float >= 5 and angle_float < 15:
                return 'right_10.mp3'
        elif angle_float >= 15 and angle_float < 25:
                return 'right_20.mp3'
        elif angle_float >= 25 and angle_float < 35:
                return 'right_30.mp3'
        elif angle_float >= 35 and angle_float < 45:
                return 'right_40.mp3'
        elif angle_float >= 45 and angle_float < 55:
                return 'right_50.mp3'





























