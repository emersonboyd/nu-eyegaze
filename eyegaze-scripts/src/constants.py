from enum import Enum


class CameraType(Enum):
    PICAM_LEFT = 1
    PICAM_RIGHT = 2
    EMERSON_IPHONE_6_PLUS = 3


class ClassType(Enum):
    EXIT_SIGN = 'exit_sign', 1
    BATHROOM_SIGN = 'bathroom_sign', 2

    def __str__(self):
        return self.value[0]

    def num(self):
        return self.value[1]


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
