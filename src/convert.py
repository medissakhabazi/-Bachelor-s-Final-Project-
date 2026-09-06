import tensorflow as tf
import hls4ml
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR , "model" , "mnist_approx_pot_logits_test.keras")
MODEL_PATH = os.path.join(BASE_DIR , "model" , "mnist_baseline_logits.keras")
OUTPUT_DIR = os.path.join(BASE_DIR , "mnist_hls_resource.keras")
PART = "xcu250-figd2104-2L-e"


REUSE_FACTOR =512


print("*" * 50)
print("Loading MNIST model")
print("*" * 50)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

model.summary()


print("\n" + "*" * 50)
print("Creating hls4ml configuration")
print("*" * 50)

config = hls4ml.utils.config_from_keras_model(
    model,
    granularity="name",
    backend="Vitis"
)



config["Model"]["Strategy"] = "Resource"


print("\n" + "*" * 50)
print("Applying layer configuration")
print("*" * 50)

for layer_name, layer_config in config["LayerName"].items():

    # Resource-oriented implementation
    if "Strategy" in layer_config:
        layer_config["Strategy"] = "Resource"

    # Reuse factor only where supported
    if "ReuseFactor" in layer_config:
        layer_config["ReuseFactor"] = REUSE_FACTOR



print("\n" + "*" * 50)
print("HLS4ML CONFIGURATION")
print("*" * 50)

for layer_name, layer_config in config["LayerName"].items():

    print(
        f"{layer_name:30s} : "
        f"Strategy={layer_config.get('Strategy', 'default')}, "
        f"ReuseFactor={layer_config.get('ReuseFactor', 'default')}"
    )

print("\n" + "*" * 50)
print("Converting Keras -> HLS")
print("*" * 50)

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir=OUTPUT_DIR,
    part=PART,
    io_type="io_parallel",
    backend="Vitis"
)

print("\nConversion completed.")
hls_model.write()


print("\n" + "*" * 50)
print("HLS CONVERSION SUCCESSFUL")
print("*" * 50)

print("Output directory:")
print(os.path.abspath(OUTPUT_DIR))

print("\nReuse Factor:")
print(REUSE_FACTOR)

print("\nStrategy:")
print("Resource")

print("\nFiles created:")

for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        print(os.path.join(root, file))

print("\nDONE.")