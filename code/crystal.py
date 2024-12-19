import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the image contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def enhance_image(image_path):
    """
    Enhance the image using CLAHE and other techniques.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE
    enhanced = apply_clahe(gray)

    # Apply GaussianBlur for smoothing (optional)
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Stack original and enhanced images for comparison
    result = np.hstack((gray, enhanced, smoothed))

    return result


def main():
    """
    Main function to handle image enhancement.
    """
    # Open file dialog to select an image
    Tk().withdraw()  # Prevent root window from appearing
    print("Please select an image file.")
    file_path = askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

    if not file_path:
        print("No file selected. Exiting...")
        return

    # Enhance the image
    enhanced_image = enhance_image(file_path)

    # Display the results
    plt.figure(figsize=(12, 8))
    plt.title("Original Image (Left), Enhanced Image (Center), Smoothed Image (Right)")
    plt.imshow(enhanced_image, cmap="gray")
    plt.axis("off")
    plt.show()

    # Save the enhanced image
    save_path = file_path.rsplit(".", 1)[0] + "_enhanced.jpg"
    cv2.imwrite(save_path, enhanced_image)
    print(f"Enhanced image saved at: {save_path}")


if __name__ == "__main__":
    main()
