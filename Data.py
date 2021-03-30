from PIL import Image
import glob
import numpy as np
import random
class Data:
    def __init__(self, path, scale_factor, batch_size, size=(256, 256)):
        self.path = path
        self.size = size
        self.scale_factor = scale_factor
        self.batch_size = batch_size

    def collect(self):

        high_quality_imgs = []
        low_quality_imgs = []
        path = glob.glob(self.path + '*.jpg')
        while(len(low_quality_imgs) < self.batch_size):
            img = Image.open(random.choice(path))
            high_quality_img = img.resize(self.size)
            low_quality_img = img.resize((self.size[0]//self.scale_factor, self.size[1]//self.scale_factor))
            high_quality_imgs.append(np.array(high_quality_img))
            low_quality_imgs.append(np.array(low_quality_img))

        high_quality_imgs = np.array(high_quality_imgs) / 127.5 - 1.
        low_quality_imgs = np.array(low_quality_imgs) / 127.5 - 1.

        return high_quality_imgs, low_quality_imgs