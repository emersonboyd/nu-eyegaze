class BoundingBox:
    def __init__(self, ymin, xmin, ymax, xmax):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def get_center_pixel(self):
        return (((self.xmax + self.xmin) / 2), ((self.ymax + self.ymin) / 2))


    def __str__(self):
        return 'x-range: {} {}, y-range: {} {}'.format(self.xmin, self.xmax, self.ymin, self.ymax)
