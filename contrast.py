import cv2
from matplotlib import pyplot as plt
import numpy as np
import os 

def calculate_contrast(image_path):
    """Calculates image contrast using RGB histograms and standard deviation.
    Args:
        image_path: Path to the image file.
    Returns:
        A dictionary containing the standard deviation of each color channel
        (Red, Green, Blue) as a measure of contrast, or None if there's an error.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or could not be loaded.")
        # Calculate histograms
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
        
        colors = ("red", "green", "blue")
        fig, ax = plt.subplots()
        ax.set_xlim([0, 256])  # Set x-axis limits for 0-255 intensity range
        for channel_id, color in enumerate(colors):
            hist = cv2.calcHist([img], [channel_id], None, [256], [0, 256])
            ax.plot(hist, color=color, label=f"histogram {color}")  # Plot the histogram

        ax.set_title("Color Histogram")
        ax.set_xlabel("Color value (0-255)")  # Clarify x-axis label
        ax.set_ylabel("Pixel count")
        plt.legend(loc="upper right")
        name = os.path.basename(image_path).split(".")[0]
      
        plt.savefig("output/hist_{}.png".format(name))
        plt.close('all')

        # Calculate standard deviations (measure of contrast)
        std_r = np.std(hist_r)
        std_g = np.std(hist_g)
        std_b = np.std(hist_b)
        contrast_measures = {
            "red": std_r,
            "green": std_g,
            "blue": std_b,
        }
        return contrast_measures
    except Exception as e:
        print(f"Error calculating contrast: {e}")
        return None
    
# Example using range as contrast measure:
def calculate_contrast_range(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or could not be loaded.")
        # Calculate histograms
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
        # Calculate range (max - min) as contrast measure
        range_r = np.max(hist_r) - np.min(hist_r)
        range_g = np.max(hist_g) - np.min(hist_g)
        range_b = np.max(hist_b) - np.min(hist_b)
        contrast_measures = {
            "red": range_r,
            "green": range_g,
            "blue": range_b,
        }
        return contrast_measures
    except Exception as e:
        print(f"Error calculating contrast: {e}")
        return None

# Example usage:
image_path = "/home/valentina/Pictures/colorcheckerclassic/before.png"  # Replace with your image path
contrast = calculate_contrast(image_path)

if contrast:
    print("Contrast Measures (Standard Deviation):")
    print(contrast)
    # Example: Average contrast
    average_contrast = np.mean(list(contrast.values()))
    print(f"Average Contrast: {average_contrast}")

# Example usage:
contrast_range = calculate_contrast_range(image_path)
if contrast_range:
    print("Contrast Measures (Range):")
    print(contrast_range)
    # Example: Average contrast
    average_contrast_range = np.mean(list(contrast_range.values()))
    print(f"Average Contrast (Range): {average_contrast_range}")
