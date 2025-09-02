# Import the functions from your other files
from preprocessor import preprocess_image
from model import create_model

import numpy as np
import tensorflow as tf
import os
import glob # Used to find all files in a folder

# --- Configuration ---
DATASET_PATH = "data/train" 
NUM_CLASSES = 2  # We'll detect this automatically later
LEARNING_RATE = 0.001
EPOCHS = 10      # How many times we train on the full dataset
BATCH_SIZE = 8   # How many images to process at once

def load_dataset(path):
    """
    
    """
    print(f"⏳ Loading and preprocessing dataset from: {path}")
    
    images = []
    labels = []
    
    # Get a list of subdirectories (e.g., ['spiderman', 'other'])
    class_names = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    if len(class_names) < 2:
        print("❌ ERROR: You need at least two subdirectories (classes) in your data/raw folder.")
        return None, None, None

    print(f"Found classes: {class_names}")

    for class_index, class_name in enumerate(class_names):
        # Find all .jpg and .png images in the class folder
        image_paths = glob.glob(os.path.join(path, class_name, '*.jpg'))
        image_paths.extend(glob.glob(os.path.join(path, class_name, '*.png')))
        
        for image_path in image_paths:
            # Preprocess the image using YOUR script
            processed_img = preprocess_image(image_path)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(class_index) # Assign the folder index as the label

    # Convert lists to NumPy arrays, which is what TensorFlow needs
    return np.array(images), np.array(labels), class_names


def main():
    """Main function to run the training pipeline."""
    # 1. Load the dataset
    X_train, y_train, class_names = load_dataset(DATASET_PATH)
    if X_train is None:
        return # Exit if dataset loading failed

    num_found_classes = len(class_names)
    print(f"\n✅ Dataset loaded successfully. Found {len(X_train)} images in {num_found_classes} classes.")

    # 2. Create the model
    model = create_model(num_classes=num_found_classes)

    # 3. Compile the model (prepare it for training)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 4. Train the model!
    print("\n🚀 Starting training...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
    print("✅ Training complete!")

    # 5. Save the trained model to a file
    model.save("image_classifier_model.h5")
    print("💾 Model saved to image_classifier_model.h5")
    # Also save the class names, so we know what the predictions mean later
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("💾 Class names saved to class_names.txt")

if __name__ == "__main__":
    main()

