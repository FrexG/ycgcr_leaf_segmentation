from ycbcr import YCBCR
from k_means import KMEANS
import os


class Main:
    def getImagePath(self, image_path, filename):
        # get path of image file
        return os.path.join(image_path, filename)


    def classifyImage(self, imagepath, filename):
        # Start Image Segmentation
        # YCBCR(filename, image=self.getImagePath(
        #    imagepath, filename))
        KMEANS(filename, image=self.getImagePath(
            imagepath, filename))


if __name__ == "__main__":
    # path to training images
    image_path = '.....'
    c = Main()
    if os.path.exists(image_path):
        for dirpath, dirname, filenames in os.walk(image_path):
            for ImageFile in filenames:
                c.classifyImage(image_path, ImageFile)
    else:
        print("Image path not found!")
