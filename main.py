import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 1. Load the saved model
model = load_model('character_recognition_model.keras')

# 2. Prepare Input Data for Prediction
def preprocess_image(img_path):
    """
    Preprocess the input image for the model.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.array: Preprocessed image ready for model prediction.
    """
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")  # Adjust target size based on model input
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Example image path for prediction (replace with actual path)
img_path = 'C:/Users/vaish/Desktop/digit1.png'
img = preprocess_image(img_path)

# 3. Make Predictions
predicted_class = np.argmax(model.predict(img), axis=-1)

# 4. Mapping Class Index to Characters
# Create a list of all characters (digits 0-9, uppercase A-Z, and lowercase a-z)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Get the predicted character
predicted_character = classes[predicted_class[0]]
print(f"Predicted character: {predicted_character}")

# 5. Run the Model (prediction will be made once the script is executed)
