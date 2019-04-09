from enum import Enum

INVALID_MEASUREMENT = "None"

class CameraType(Enum):
    # 0: CAM_NUMBER
    # 1: PIXEL_SIZE_MICROMETERS
    # 2: FOCAL_LENGTH_MILLIMETERS
    # 3: BASELINE_MILLIMETERS
    # 4: HORIZONTAL_FIELD_OF_VIEW_DEGREES
    # 5: VERTICAL_FIELD_OF_VIEW_DEGREES
    PICAM_LEFT = 1, 1.12, 3.04, 116.586, 62.2, 48.8
    PICAM_RIGHT = 2, 1.12, 3.04, 116.586, 62.2, 48.8
    EMERSON_IPHONE_6_PLUS = 3, 1.5, 4.15, 167, 73 * 3/4, 73

    def get_pixel_size(self):
        return self.value[1]

    def get_focal_length(self):
        return self.value[2]

    def get_baseline(self):
        return self.value[3]

    def get_horizontal_field_of_view(self):
        return self.value[4]

    def get_vertical_field_of_view(self):
        return self.value[5]


class ClassType(Enum):
    EXIT_SIGN = 'exit_sign', 1, 'exit_sign.mp3'
    BATHROOM_SIGN = 'bathroom_sign', 2, 'bathroom_sign.mp3'

    def __str__(self):
        return self.value[0]

    def num(self):
        return self.value[1]

    def get_audio_file_name(self):
        return self.value[2]


def get_class_type_for_number(n):
    for c in ClassType:
        if c.value[1] == n:
            return c


def get_class_type_for_string(s):
    for c in ClassType:
        if c.value[0] == s:
            return c


if __name__ == '__main__':
    e = ClassType.EXIT_SIGN
    print(e)
