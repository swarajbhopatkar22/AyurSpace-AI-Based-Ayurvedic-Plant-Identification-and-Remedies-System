from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess single image for prediction (same as training)."""
    img = load_img(image_path, target_size=target_size)   # PIL image
    x = img_to_array(img)                                # to numpy
    x = np.expand_dims(x, axis=0)                        # (1,224,224,3)
    x = preprocess_input(x)                              # ResNet50 style
    return x


def create_datagenerator(data_dir):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.25,
        horizontal_flip=True
    )


    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen


if __name__ == "__main__":
    print("Data preprocessing ready!")
