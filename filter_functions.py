import numpy as np
from utils.rgb2gray import rgb2gray_approx, rgb2gray

def inverse_image(image):
    return np.invert(image)

def grayscale_image(image):
    #print(image.shape, type(image), image.dtype, "min, mean, max", np.min(image), np.mean(image), np.max(image))
    gray_1ch = rgb2gray_approx(image).astype(np.uint8)
    return np.stack((gray_1ch,)*3, axis=-1) # back to to 3 channels

def grayscale_image_proper_but_slow(image): # kept here for easy visualization of how it works...    
    gray_1ch = rgb2gray(image).astype(np.uint8) # actually pretty slow!
    return np.stack((gray_1ch,)*3, axis=-1) # back to to 3 channels


class Filters_Handler(object):
    """
    Remembers which filters are on and also applies them.
    (note: these could also be chained to preserve the order ... maybe if that's desired?)
    """

    def __init__(self):
        self.grayscale = False
        self.grayscale_slow = False
        self.inverse = False

    def num_to_filter(self, number):
        if number == 1:
            print("1 = Inversion")
            self.inverse = not self.inverse
        elif number == 2:
            print("2 = Grayscale")
            self.grayscale = not self.grayscale

        elif number == 9: ### kept at the end, only for properness of bw transforms ...
            print("9 = Grayscale (slow)")
            self.grayscale_slow = not self.grayscale_slow

        else:
            print("Number", number,"doesn't have coded filter. (Note: to turn of Filter controls use 'f')")

    def apply_all_filters(self, image):
        #print(image.shape, type(image), image.dtype, "min, mean, max", np.min(image), np.mean(image), np.max(image))

        # << Order: >>
        # - grayscale
        # - inverse

        if self.grayscale:
            image = grayscale_image(image)
        if self.grayscale_slow:
            image = grayscale_image_proper_but_slow(image)
        if self.inverse:
            image = inverse_image(image)

        return image