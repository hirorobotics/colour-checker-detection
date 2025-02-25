import glob
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import colour
import imageio.v2 as imageio
from colour_checker_detection import (
    ROOT_RESOURCES_EXAMPLES,
    detect_colour_checkers_segmentation)

def center_crop(img, dim):
  """Returns center cropped image

  Args:
  img: image to be center cropped
  dim: dimensions (width, height) to be cropped from center
  """
  width, height = img.shape[1], img.shape[0]
  #process crop width and height for max available dimension
  crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
  crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
  mid_x, mid_y = int(width/2), int(height/2)
  cw2, ch2 = int(crop_width/2), int(crop_height/2) 
  crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  return crop_img

colour.plotting.colour_style()

colour.utilities.describe_environment()
##images
# COLOUR_CHECKER_IMAGE_PATHS = glob.glob(
#    #os.path.join(ROOT_RESOURCES_EXAMPLES, 'colour-checker-detection-dataset/train/images', '*19*.png'))
#     os.path.join(ROOT_RESOURCES_EXAMPLES, 'detection', '*1967.png'))
path = "/home/valentina/Pictures/colorcheckerclassic/after.png"
# name = os.path.basename(COLOUR_CHECKER_IMAGE_PATHS[0]).split(".")[0]
filename, file_extension = os.path.splitext(path)
filepath = path.replace(file_extension, ".txt")
path2 = path.replace(filename, filename + "2")
im = imageio.imread(path)
crop_img = center_crop(im, (1000,1000))
imageio.imwrite(path2, crop_img, format=None) 
#COLOUR_CHECKER_IMAGE_PATHS = glob.glob(
#       os.path.join( '/home/valentina/Documents/python-macduff-colorchecker-detector/examples', 'test.jpg'))
COLOUR_CHECKER_IMAGE_PATHS = glob.glob(path2)
COLOUR_CHECKER_IMAGES = [
    colour.cctf_decoding(colour.io.read_image(path))
    for path in COLOUR_CHECKER_IMAGE_PATHS
]
###Colour Fitting
D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
    'ColorChecker24 - After November 2014']

colour_checker_rows = REFERENCE_COLOUR_CHECKER.rows
colour_checker_columns = REFERENCE_COLOUR_CHECKER.columns

# NOTE: The reference swatches values as produced by the "colour.XYZ_to_RGB"
# definition are linear by default.
# See https://github.com/colour-science/colour-checker-detection/discussions/59
# for more information.
REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        'sRGB', REFERENCE_COLOUR_CHECKER.illuminant)

height, width, channel = 64, 64, 3

for image in COLOUR_CHECKER_IMAGES:
    colour.plotting.plot_image(colour.cctf_encoding(image))
### detection
SWATCHES = []
image_color = []
masks_color = []
for image in COLOUR_CHECKER_IMAGES:
    for colour_checker_data in detect_colour_checkers_segmentation(
        image, additional_data=True):
        
        swatch_colours, swatch_masks, colour_checker_image, _ = (
            colour_checker_data.values)
        SWATCHES.append(swatch_colours)
        image_color = colour_checker_image
        masks_color = swatch_masks
        
        # Using the additional data to plot the colour checker and masks.
        masks_i = np.zeros(colour_checker_image.shape)
        for i, mask in enumerate(swatch_masks):
            masks_i[mask[0]:mask[1], mask[2]:mask[3], ...] = 1
        
        colour.plotting.plot_image(
            colour.cctf_encoding(
                np.clip(colour_checker_image + masks_i * 0.25, 0, 1)))

colour.plotting.plot_multi_colour_swatches(REFERENCE_SWATCHES, columns=6, direction ="-y")
colour.plotting.plot_multi_colour_swatches(SWATCHES[0], columns=6, direction ="-y")
# for reference_swatches in REFERENCE_SWATCHES.tolist():  
#     # step-4 Generate RGB Numpy Array 
#     red, green, blue = reference_swatches
#     img = np.full((height, width, channel), [red, green, blue], dtype=('float32'))
#     colour.plotting.plot_image(
#             colour.cctf_encoding(
#                 np.clip(img)))
    
# for color_swatches in SWATCHES[0].tolist():  
#     # step-4 Generate RGB Numpy Array 
#     red, green, blue = color_swatches
#     img = np.full((height, width, channel), [red, green, blue], dtype=('float32'))
#     colour.plotting.plot_image(
#             colour.cctf_encoding(
#                 np.clip(img)))  
corrected_swatches = []
for i, swatches in enumerate(SWATCHES):
    swatches_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
        swatches, 'sRGB', D65))
    
    colour_checker = colour.characterisation.ColourChecker(
        os.path.basename(COLOUR_CHECKER_IMAGE_PATHS[i]),
        dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_xyY)),
        D65, colour_checker_rows, colour_checker_columns)
    
    colour.plotting.plot_multi_colour_checkers(
        [REFERENCE_COLOUR_CHECKER, colour_checker])

    swatches_f = colour.colour_correction(swatches, swatches, REFERENCE_SWATCHES)
    corrected_swatches = swatches_f
    swatches_f_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(
        swatches_f, 'sRGB', D65))
    colour_checker = colour.characterisation.ColourChecker(
        '{0} - CC'.format(os.path.basename(COLOUR_CHECKER_IMAGE_PATHS[i])),
        dict(zip(REFERENCE_COLOUR_CHECKER.data.keys(), swatches_f_xyY)),
        D65, colour_checker_rows, colour_checker_columns)
    
    colour.plotting.plot_multi_colour_checkers(
        [REFERENCE_COLOUR_CHECKER, colour_checker])

    colour.plotting.plot_image(colour.cctf_encoding(
        colour.colour_correction(
            COLOUR_CHECKER_IMAGES[i], swatches, REFERENCE_SWATCHES)))

errors_color_wrong = np.zeros((np.shape(SWATCHES[0])[0], 3), dtype=float)
errors_color = np.zeros((np.shape(SWATCHES[0])[0], 3), dtype=float)
imageplot = image_color
masksplot = masks_color
colors = ("red", "green", "blue")
colors2 = ("magenta", "chartreuse", "aqua")
for m in range(np.shape(SWATCHES[0])[0]):
    maskplot = masksplot[m]
    i=imageplot[maskplot[0] : maskplot[1], maskplot[2] : maskplot[3], ...]
    # fig, ax = plt.subplots()
    # ax.set_xlim([0.0, 1.0])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
                i[:, :, channel_id], bins=256, range=(0, 1)
            )
        # ax.plot(bin_edges[0:-1], histogram, color=color, label="histogram patch {}".format(color))
        # ax.axvline(SWATCHES[0][m, channel_id], color=color, linestyle='--', label="mean {}".format(color))
        # ax.axvline(corrected_swatches[m, channel_id], color=colors2[channel_id], label="corrected {}".format(color))
        # ax.axvline(REFERENCE_SWATCHES[m, channel_id], color=color, label="reference color {}".format(color))
        error = np.abs(REFERENCE_SWATCHES[m, channel_id]-corrected_swatches[m, channel_id])
        errors_color[m, channel_id] = error
        error_wrong = (REFERENCE_SWATCHES[m, channel_id]-SWATCHES[0][m, channel_id])
        errors_color_wrong[m, channel_id] = error_wrong
    # ax.set_title("Color Histogram")
    # ax.set_xlabel("Color value")
    # ax.set_ylabel("Pixel count")
    # plt.legend(loc="upper right")
    # plt.savefig("output/hist_correction_{}.png".format(m))
    # plt.close('all')

mean_error_wrong = np.mean(errors_color_wrong, axis=0) 
mean_error = np.mean(errors_color, axis=0) 
std_error = np.std(errors_color, axis=0) 
std_error_wrong = np.std(errors_color_wrong, axis=0) 

fig, ax = plt.subplots()
ax.set_xlim([0.0, 1.0])
for channel_id, color in enumerate(colors):
    histogram, bin_edges = np.histogram(
            errors_color_wrong[:, channel_id], bins=256, range=(-1, 1)
        )
    ax.plot(bin_edges[0:-1], histogram, color=color, label="histogram error on {}".format(color))
    ax.axvline(mean_error_wrong[channel_id], color=color, linestyle='--', label="mean {}".format(color))
ax.set_title("error Histogram")
ax.set_xlabel("error value")
ax.set_ylabel("Pixel count")
plt.legend(loc="upper right")
plt.savefig("output/Error_{}.png".format("3 channels"))
plt.close('all')
f = open(filepath, "a")

for channel_id, color in enumerate(colors):
    stringtosave = "mean error {} is {} std is {} \n".format(color, mean_error_wrong[channel_id], std_error_wrong[channel_id])
    print(stringtosave)  
    f.write(stringtosave)
f.close()

# ###Additional Data Plotting
# for image in COLOUR_CHECKER_IMAGES:
#     for colour_checker_data in detect_colour_checkers_segmentation(
#             image, show=True):
#         pass
# ###Failures
# COLOUR_CHECKER_IMAGE_PATHS = glob.glob(
#     #os.path.join(ROOT_RESOURCES_EXAMPLES, "colour-checker-detection-dataset/train/images", "*25*.png"))
#      #os.path.join(ROOT_RESOURCES_EXAMPLES, "detection", "*25*.png"))
# os.path.join("/home/valentina/Documents/python-macduff-colorchecker-detector/examples/below_fishbox.jpg"))
# for path in COLOUR_CHECKER_IMAGE_PATHS:
#     for colour_checker_data in detect_colour_checkers_segmentation(
#         path, apply_cctf_decoding=True, show=True
#     ):
#         pass

