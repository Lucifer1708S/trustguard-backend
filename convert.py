import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# 1. Rebuild the exact same brain structure
img_size = 128
base_model = EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

# 2. Load the weights you just saved
model.load_weights("final_weights.weights.h5")
print("✅ Brain rebuilt and weights loaded!")

# 3. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("🚀 model.tflite is ready! Upload it to Render NOW!")