# Source: https://towardsdatascience.com/ganscapes-using-ai-to-create-new-impressionist-paintings-d6af1cf94c56
# A mild contrast adjustment and image resizing, to restore the original aspect ratio
# - The first part of the code converts the image to floating-point, finds the 0.1% min and the 0.99% max of each channel, and scales up the contrast. This is akin to Adjust Levels feature in Photoshop.

import numpy as np
import PIL

#from numba import jit # ... some functions unsupported

def auto_contrast(image):
    # convert the image to use floating point
    img_fp = image.astype(np.float32)
    # stretch the red channel by 0.1% at each end
    r_min = np.percentile(img_fp[:,:,0:1], 0.1)
    r_max = np.percentile(img_fp[:,:,0:1], 99.9)
    img_fp[:,:,0:1] = (img_fp[:,:,0:1]-r_min) * 255.0 / (r_max-r_min)
    # stretch the green channel by 0.1% at each end
    g_min = np.percentile(img_fp[:,:,1:2], 0.1)
    g_max = np.percentile(img_fp[:,:,1:2], 99.9)
    img_fp[:,:,1:2] = (img_fp[:,:,1:2]-g_min) * 255.0 / (g_max-g_min)
    # stretch the blue channel by 0.1% at each end
    b_min = np.percentile(img_fp[:,:,2:3], 0.1)
    b_max = np.percentile(img_fp[:,:,2:3], 99.9)
    img_fp[:,:,2:3] = (img_fp[:,:,2:3]-b_min) * 255.0 / (b_max-b_min)
    # convert the image back to integer, after rounding and clipping
    image = np.clip(np.round(img_fp), 0, 255).astype(np.uint8) # < np.clip only from numba 0.54
    return image