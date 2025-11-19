import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
import os

train_dir = "data/train"
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=30,
    label_mode="categorical"
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),
    batch_size=30,
    label_mode="categorical"
)

normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=50, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print("Accuracy:", test_acc)

img = image.load_img("image/1.jpg", target_size=(128, 128))
arr = image.img_to_array(img) / 255.0
arr = np.expand_dims(arr, 0)

pred = model.predict(arr)
idx = np.argmax(pred[0])

print("Імовірності:", pred[0])
print("Модель вирішила:", class_names[idx])
