import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image, resizes it, and normalizes its pixel values.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target (width, height) for resizing.

    Returns:
        numpy.ndarray: The processed image as a NumPy array, or None if failed.
    """
    # 1. Check if the file exists
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image not found at path: {image_path}")
        return None

    # 2. Load the image using OpenCV
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ ERROR: OpenCV could not read the image. It might be corrupted or in an unsupported format.")
        return None

    # 3. Resize the image
    resized_image = cv2.resize(original_image, target_size)

    # 4. Normalize the pixel values from [0, 255] to [0.0, 1.0]
    # We convert the image to a floating-point data type before dividing
    normalized_image = resized_image.astype(np.float32) / 255.0

    print(f"✅ Successfully preprocessed image: {image_path}")
    return normalized_image

# This block allows you to test the function directly by running this script
if __name__ == "__main__":
    print("--- Starting Preprocessor Test ---")

    # IMPORTANT: Place your test image in the 'data/raw/' folder
    # Let's use your Spider-Man image. Make sure the filename matches.
    test_image_filename = "spider man lock screen.jpg"
    test_image_path = os.path.join("data", "raw", test_image_filename)

    # Run the preprocessing function
    processed_image = preprocess_image(test_image_path)

    if processed_image is not None:
        print("\nProcessing successful!")
        # Print some details about the output
        print(f"Output image shape: {processed_image.shape}")
        print(f"Output data type: {processed_image.dtype}")
        print(f"Min pixel value: {np.min(processed_image):.2f}")
        print(f"Max pixel value: {np.max(processed_image):.2f}")
    else:
        print("\nProcessing failed. Please check the error messages above.")

    print("--- Test Finished ---")
