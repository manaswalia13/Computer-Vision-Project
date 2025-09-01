import tensorflow as tf
import numpy as np
from preprocessor import preprocess_image # We reuse your preprocessor!
import sys

# --- Configuration ---
MODEL_PATH = "image_classifier_model.h5"
CLASS_NAMES_PATH = "class_names.txt"

def load_prediction_tools(model_path, class_names_path):
    """Loads the trained model and the class names."""
    try:
        model = tf.keras.models.load_model(model_path)
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        print("✅ Model and class names loaded successfully.")
        return model, class_names
    except Exception as e:
        print(f"❌ ERROR loading tools: {e}")
        return None, None

def predict_image(model, class_names, image_path):
    """
    Takes a path to a new image, preprocesses it, and predicts its class.
    """
    # 1. Preprocess the image using the same function from training
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return

    # 2. The model expects a "batch" of images, so we add an extra dimension
    image_batch = np.expand_dims(processed_image, axis=0)

    # 3. Make the prediction
    predictions = model.predict(image_batch)
    
    # 4. Interpret the results
    # The output 'predictions' is a list of probabilities for each class.
    # We find the class with the highest probability.
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100 # Convert to percentage
    predicted_class_name = class_names[predicted_class_index]

    print("\n--- 🤖 Prediction Result ---")
    print(f"I am {confidence:.2f}% confident that this is an image of class: '{predicted_class_name}'")
    print("--------------------------")

if __name__ == "__main__":
    # Load the tools once
    model, class_names = load_prediction_tools(MODEL_PATH, CLASS_NAMES_PATH)

    if model is not None:
        # Check if the user provided an image path as an argument
        if len(sys.argv) > 1:
            image_to_predict = sys.argv[1]
            predict_image(model, class_names, image_to_predict)
        else:
            print("\n👉 USAGE: Please provide the path to an image.")
            print("Example: python predict.py data/raw/img1/spider_man_lock_screen.jpg")
