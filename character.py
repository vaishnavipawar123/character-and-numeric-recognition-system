import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Set up paths
data_dir = "characters"
img_height, img_width = 28, 28  # Adjust based on your image size
batch_size = 32
num_classes = 62  # 10 digits + 26 uppercase letters + 26 lowercase letters

# Label mapping for your dataset
def get_label_from_filename(filename):
    file_num = int(filename.split('-')[0].replace('img', ''))
    if file_num <= 10:
        return file_num - 1  # 0-9 digits
    elif file_num <= 36:
        return file_num - 11 + 10  # A-Z uppercase letters
    elif file_num <= 62:
        return file_num - 37 + 36  # a-z lowercase letters
    else:
        raise ValueError("Filename not in expected range")

# Data generator
def custom_data_generator(data_dir, batch_size):
    filenames = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".png"):
                label = get_label_from_filename(file)
                file_path = os.path.join(root, file)
                filenames.append(file_path)
                labels.append(label)
    
    while True:
        indices = np.arange(len(filenames))
        np.random.shuffle(indices)
        for i in range(0, len(filenames), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = []
            batch_labels = []
            for idx in batch_indices:
                img = tf.keras.preprocessing.image.load_img(filenames[idx], target_size=(img_height, img_width), color_mode='grayscale')
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                batch_images.append(img_array)
                batch_labels.append(labels[idx])
            yield np.array(batch_images), to_categorical(batch_labels, num_classes=num_classes)

# Model creation
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
input_shape = (img_height, img_width, 1)
model = create_model(input_shape, num_classes)

# Training the model
epochs = 10
steps_per_epoch = len(os.listdir(data_dir)) // batch_size
train_gen = custom_data_generator(data_dir, batch_size)

model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Save the model
model.save("character_recognition_model.keras")

