import tensorflow as tf

def create_model(num_classes):
    """
    Loads a pre-trained MobileNetV2 model and prepares it for our custom task.

    Args:
        num_classes (int): The number of categories we want to classify (e.g., 2 for cats and dogs).

    Returns:
        A Keras model object.
    """
    # Load the MobileNetV2 model, using pre-trained weights from 'imagenet'.
    # We set `include_top=False` to remove the original final layer.
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    # Freeze the layers of the base model. We don't want to change them during training.
    base_model.trainable = False

    # Now, we add our own custom layers on top of the base model.
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # This layer condenses the features.
        tf.keras.layers.Dropout(0.2),             # Helps prevent overfitting.
        tf.keras.layers.Dense(num_classes, activation='softmax') # Our final output layer.
    ])

    print("✅ Model created successfully.")
    return model

# This block allows you to test the function directly by running `python model.py`
if __name__ == "__main__":
    # Let's assume we are classifying 2 different types of objects (e.g., Spider-Man vs. someone else)
    NUM_CATEGORIES = 2
    my_model = create_model(NUM_CATEGORIES)
    
    # Print a summary of the model's architecture to see what you've built
    my_model.summary()

