import cv2
import numpy as np
import matplotlib.pyplot as plt


def increase_contrast_clahe_rgb(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) for RGB images.

    Args:
        image: The input RGB image.
        clip_limit: Clipping limit.
        tile_grid_size: Tile grid size.

    Returns:
        The contrast-adjusted RGB image.
    """
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    yuv_image[:, :, 0] = clahe.apply(yuv_image[:, :, 0])  # Equalize Y channel
    new_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return new_image

def sigmoid(x, factor, center):
    scale = factor*50
    x = np.asarray(x)
    result = 1/(1 + np.exp(-(x-center)*scale))
    return result

def contrast(image, contrast_factor=1.0, midpoint=0.5):
    """
    Applies an S-curve contrast adjustment to an image.

    Args:
        image (numpy.ndarray): The input image (grayscale or color).
        contrast_factor (float): Controls the steepness of the S-curve. 
                                  Values < 1 increase contrast, > 1 decrease.
        midpoint (float): The point of inflection of the S-curve (0.0-1.0).
                          Represents the pixel value that remains unchanged.

    Returns:
        numpy.ndarray: The contrast-adjusted image.  Returns the original
                       image if there's an issue with the input type.
    """

    if not isinstance(image, np.ndarray):
        print("Error: Input image must be a NumPy array.")
        return image  # Or raise an exception if you prefer

    # Ensure the image is float type for calculations
    image_float = image.astype(np.float32) / 255.0

    # Clip values to be within 0-1 range (important for the curve)
    image_float = np.clip(image_float, 0.0, 1.0)
    x = np.linspace(0, 1)
    y = sigmoid(x, contrast_factor, midpoint)
    # Create the plot
    plt.plot(x, y)
    plt.title(f"S-curve Contrast Adjustment (contrast_factor={contrast_factor}, midpoint={midpoint})")
    plt.xlabel("Input Pixel Value")
    plt.ylabel("Output Pixel Value")
    plt.grid(True)
    plt.show()
    adjusted_image = sigmoid(image_float, contrast_factor, midpoint)
    # Convert back to uint8 and scale to 0-255 for display/saving
    adjusted_image = (np.clip(adjusted_image, 0.0, 1.0) * 255).astype(np.uint8)

    return adjusted_image

def adjust_gamma(image, gamma=1.0):
    """Builds a lookup table mapping pixel values to their adjusted values."""
    invGamma = 1.0 / gamma
    table = np.array([((val / 255.0) ** invGamma) * 255
                      for val in range(0, 256)], dtype="uint8")
    return cv2.LUT(image, table)

# Example Usage:
image = cv2.imread("/home/valentina/Pictures/correction/Image__2025-02-06__14-19-54.png")  # Load as color image

if image is None:
    print("Error: Could not read image.")
    exit()

import matplotlib
matplotlib.use('qtagg')  
# im = cv2.imread("/home/valentinasanguineti/Downloads/Image__2025-02-06__14-19-54.png")  # BGR
# im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##gamma
gamma = 2.1  # Gamma < 1: Brighter image; Gamma > 1: Darker image. Experiment!
im_gamma = adjust_gamma(im_gr, gamma=gamma)
# plt.imshow(im_gamma, cmap='gray')
# plt.title(f"Gamma Corrected (gamma = {gamma})")
# plt.show()
##partial equalization
# im_equalized = cv2.equalizeHist(im_gamma)
# alpha = 0.5  # Blend factor (0.0 - 1.0). Closer to 0: Original image. Closer to 1: Fully equalized.
# im_gamma_eq = cv2.addWeighted(im_gamma, 1 - alpha, im_equalized, alpha, 0)
# plt.imshow(im_gamma, cmap='gray')
# plt.title("Gamma Image")
# plt.show()

contrasted_gray = contrast(im_gamma, contrast_factor=0.2, midpoint=0.5) # Example parameters
# cv2.imshow("Original Gray", im_gr)
# cv2.imshow("gamma Gray", im_gamma)
# cv2.imshow("Contrasted Gray", contrasted_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("/home/valentina/Pictures/correction/contrasted_gray_image_gamma21contrast02.jpg", contrasted_gray) #Save the image

print("")

# # CLAHE
# contrast_adjusted_clahe = increase_contrast_clahe_rgb(image, clip_limit=3.0)