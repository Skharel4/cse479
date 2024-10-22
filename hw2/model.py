import tensorflow as tf



def vanilla_model(class_count=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(class_count, activation="softmax")
    ])
    return model

def strawberry(class_count=100):
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.3),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.4),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.5),

        # Fully Connected Layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # Output Layer
        tf.keras.layers.Dense(class_count, activation='softmax')
    ])
    return model

