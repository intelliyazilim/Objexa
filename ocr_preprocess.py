import cv2

class HeightImageResizer:

    def __init__(self, target_width=600, resample_method=cv2.INTER_LANCZOS4):
        self.target_width = target_width
        self.resample_method = resample_method

    def resize(self, image):
        h, w = image.shape[:2]
        scale_factor = self.target_width / w
        new_height = int(h * scale_factor)
        return cv2.resize(image, (self.target_width, new_height), interpolation=self.resample_method)

class SpeedImageResizer:

    def __init__(self, target_size=(144, 144), resample_method=cv2.INTER_LINEAR):
        self.target_size = target_size
        self.resample_method = resample_method

    def resize(self, image):
        return cv2.resize(image, self.target_size, interpolation=self.resample_method)
