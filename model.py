import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from preprocess import create_datagenerator
import matplotlib.pyplot as plt

def create_model(num_classes=4):
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model():
    train_gen, val_gen = create_datagenerator('dataset/')

    print("Class indices:", train_gen.class_indices)

    model = create_model(num_classes=len(train_gen.class_indices))

    # callbacks add karo
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=40,              # yahan 40
        callbacks=[early_stop, lr_reduce]
    )

    model.save('models/ayurspace_model.h5')
    print("Model saved as 'models/ayurspace_model.h5'")

    return history


if __name__ == "__main__":
    train_model()
    print("Training complete!")