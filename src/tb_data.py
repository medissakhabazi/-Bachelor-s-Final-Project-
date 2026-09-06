import os
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR , "model" , "mnist_baseline_logits.keras")
MNIST_PATH = os.path.join( BASE_DIR , "data" , "mnist.npz")

N = 10000

INPUT_FILE ="tb_input_features.dat"
OUTPUT_FILE ="tb_output_predictions.dat"



print("*" * 50)
print("Loading Keras model")
print("*" * 50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

model.summary()




print()
print("*" * 50)
print("Loading local MNIST")
print("*" * 50)

if not os.path.exists(MNIST_PATH):
    raise FileNotFoundError(
        f"MNIST dataset not found:\n{MNIST_PATH}\n"
    )

print("Dataset path:", MNIST_PATH)

with np.load(MNIST_PATH) as data:
    x_test = data["x_test"]
    y_test = data["y_test"]

x_test = x_test.astype(np.float32) / 255.0

x = x_test[:N]
y = y_test[:N]

print("MNIST input shape:", x.shape)
print("MNIST labels shape:", y.shape)


if len(model.input_shape) == 4:

   
    x_model = np.expand_dims(
        x,
        axis=-1
    )

else:

   

    x_model = x


print("Input given to model:", x_model.shape)



print()
print("=" * 70)
print("Running Keras prediction")
print("=" * 70)

pred = model.predict(
    x_model,
    verbose=0
)

print("Prediction shape:", pred.shape)

pred_class = np.argmax(
    pred,
    axis=1
)

accuracy = np.mean(
    pred_class == y
)

print()
print("First 10 true labels:")
print(y[:10])

print()
print("First 10 predicted labels:")
print(pred_class[:10])


x_flat = x_model.reshape(
    N,
    -1
)

print()
print("HLS input shape:", x_flat.shape)
print("HLS output shape:", pred.shape)


np.savetxt(
    INPUT_FILE,
    x_flat,
    fmt="%.8f"
)


np.savetxt(
    OUTPUT_FILE,
    pred,
    fmt="%.8f"
)

print()
print("*" * 50)
print("FILES CREATED")
print("*" * 50)

print("Input file :", INPUT_FILE)
print("Output file:", OUTPUT_FILE)

print()
print("First input values:")
print(x_flat[0][:10])

print()
print("First prediction (10 logits):")
print(pred[0])

print()
print("Predicted class:")
print(np.argmax(pred[0]))

print()
print("True class:")
print(y[0])

print()
print("DONE.")