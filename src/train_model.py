import os
# Disable XLA temporarily
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPU found: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
    except RuntimeError:
        pass

else:
    print("GPU is NOT available. Training will run on CPU.")

print("*" * 50)
print("Loading MNIST dataset")
print("*" * 50)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mnist.npz"
)

print(BASE_DIR)
print(DATA_PATH)
print(os.path.exists(DATA_PATH))


if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "\nMNIST file not found.\n\n"
        f"Expected path:\n{DATA_PATH}\n\n"
    )
try:

    with np.load(DATA_PATH) as data:

        required_keys = {
            "x_train",
            "y_train",
            "x_test",
            "y_test"
        }

        if not required_keys.issubset(set(data.files)):

            raise ValueError(
                "Invalid mnist.npz structure.\n"
                "Required keys:\n"
                "x_train, y_train, x_test, y_test"
            )

        x_train = data["x_train"]
        y_train = data["y_train"]

        x_test = data["x_test"]
        y_test = data["y_test"]


except Exception as e:

    raise RuntimeError(
        f"\nError while reading MNIST:\n{e}"
    ) from e

print(f"Train images: {x_train.shape}")
print(f"Train labels: {y_train.shape}")
print(f"Test images : {x_test.shape}")
print(f"Test labels : {y_test.shape}")

print()
print("*" * 50)
print("Preprocessing")
print("*" * 50)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


print()
print("=" * 70)
print("Creating MNIST model")
print("=" * 70)

model = models.Sequential([

    layers.Input(shape=(28, 28)),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.2),

    layers.Dense(
        10
    )
])

print()
print("*" * 50)
print("MODEL SUMMARY")
print("*" * 50)

model.summary()

print()
print("*" * 50)
print("Compiling model")
print("*" * 50)

model.compile(

    optimizer="adam",

    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),

    metrics=["accuracy"]
)

print()
print("*" * 50)
print("Starting training process")
print("*" * 50)

history = model.fit(

    x_train,

    y_train,

    epochs=50,

    batch_size=128,

    validation_split=0.1,

    verbose=1
)

print()
print("*" * 50)
print("Evaluating model")
print("*" * 50)

test_loss, test_acc = model.evaluate(

    x_test,

    y_test,

    batch_size=128,

    verbose=2
)


print()
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")

print()
print("*" * 50)
print("Testing prediction")
print("*" * 50)

logits = model.predict(
    x_test[:10],
    verbose=0
)

predictions = np.argmax(
    logits,
    axis=1
)

print("True labels :")
print(y_test[:10])

print("Predictions  :")
print(predictions)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mnist_baseline_logits.keras"
)

model.save(MODEL_PATH)

print()
print("*" * 50)
print("MODEL SAVED")
print("*" * 50)

print()
print("Model path:")
print(MODEL_PATH)

print()
print("Training completed successfully.")