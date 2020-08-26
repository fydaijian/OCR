import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import random
import cv2

# img = cv2.imread(r'E:\deep_learning\detect_face\datasets\lfwdata\BRR\01.jpg')
# print(type(img))
# print(isinstance(img, np.ndarray))

class image_distort:

    def __init__(self):
        #image的格式是PIL
        #PIL->array： img = np.array(img)
        #array->PIL:  img = Image.fromarray(img.astype('uint8')).convert("RGB")

        self.image_distort_strategy = {
            "expand_prob": 0.5,
            "expand_max_ratio": 2,
            "hue_prob": 0.5,
            "hue_delta": 18,
            "contrast_prob": 0.5,
            "contrast_delta": 0.5,
            "saturation_prob": 0.5,
            "saturation_delta": 0.5,
            "brightness_prob": 0.5,
            "brightness_delta": 0.125
        }

    def resize_img(self, img, input_size):

        target_size = input_size
        percent_h = float(target_size[1] / img.size[1])
        percent_w = float(target_size[2] / img.size[0])
        percent = min(percent_h, percent_w)
        resized_width = int(round(img.size[0] * percent))
        resized_height = int(round(img.size[1] * percent))
        w_off = (target_size[2] - resized_width) / 2
        h_off = (target_size[1] - resized_height) / 2
        img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
        array = np.array((target_size[1], target_size[2], 3), np.uint8)
        array[:, :, 0] = 127
        array[:, :, 1] = 127
        array[:, :, 2] = 127
        ret = Image.fromarray(array)
        ret.paste(img, (np.random.randint(0, w_off + 1), int(h_off)))

        return ret

    def random_brightness(self, img):

        prob = np.random.uniform(0, 1)
        if prob < self.image_distort_strategy['brightness_prob']:
            brightness_delta = self.image_distort_strategy['brightness_delta']
            delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
            img = ImageEnhance.Brightness(img).enhance(delta)

        return img

    def random_contrast(self, img):

        prob = np.random.uniform(0, 1)
        if prob < self.image_distort_strategy['saturation_prob']:
            saturation_delta = self.image_distort_strategy['saturation_delta']
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Contrast(img).enhance(delta)

        return img

    def random_saturation(self, img):
        prob = np.random.uniform(0, 1)
        if prob < self.image_distort_strategy['saturation_prob']:
            saturation_delta = self.image_distort_strategy['saturation_delta']
            delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
            img = ImageEnhance.Color(img).enhance(delta)

        return img

    def random_hue(self, img):

        prob = np.random.uniform(0, 1)
        if prob < self.image_distort_strategy['hue_prob']:
            hue_delta = self.image_distort_strategy['hue_delta']
            delta = np.random.uniform(-hue_delta, hue_delta)
            img_hsv = np.array(img.convert('HSV'))
            img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
            img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')

        return img

    def distort_image(self, img):

        prob = np.random.uniform(0, 1)
        if prob > 0.5:
            img = self.random_brightness(img)
            img = self.random_contrast(img)
            img = self.random_saturation(img)
            img = self.random_hue(img)

        else:
            img = self.random_brightness(img)
            img = self.random_saturation(img)
            img = self.random_hue(img)
            img = self.random_contrast(img)

        return img

if __name__ == "__main__":

    img = Image.open(r'E:\deep_learning\detect_face\datasets\lfwdata\BRR\01.jpg')
    img = image_distort().distort_image(img)
    img.show()





