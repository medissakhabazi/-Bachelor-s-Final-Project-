import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR , "model" , "mnist_baseline_logits.keras")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR , "model" , "mnist_approx_pot_logits_test.keras")

MNIST_PATH = os.path.join( BASE_DIR , "data" , "mnist.npz")

SELECTED_THRESHOLD = 0.10

ALLOWED_EXPONENTS = {-3, -2, -1, 0, 1, 2, 3, 4}


def apply_pot_to_array(weights, prune_threshold):
    

    weights = np.asarray(weights, dtype=np.float32)


    w_safe = np.where(weights == 0, 1e-9, weights)

    
    abs_w = np.abs(w_safe)

    
    power = np.round(np.log2(abs_w)).astype(np.int32)

    
    nearest_pot = np.power(2.0, power).astype(np.float32)

   
    approx_weights = np.sign(weights) * nearest_pot

    
    final_weights = np.where(
        abs_w <= prune_threshold,
        0.0,
        approx_weights
    )

    
    final_weights = np.where(
        weights == 0,
        0.0,
        final_weights
    )

    return final_weights.astype(np.float32)



def analyze_pot_exponents(weights, layer_name=""):

    nonzero = weights[weights != 0]

    if nonzero.size == 0:
        print(f"\n[{layer_name}] No non-zero weights.")
        return

    exponents = np.round(
        np.log2(np.abs(nonzero))
    ).astype(np.int32)

    unique_exp, counts = np.unique(
        exponents,
        return_counts=True
    )

    print("\n" + "=" * 60)
    print(f"PoT exponent distribution: {layer_name}")
    print("=" * 60)

    for exp, count in zip(unique_exp, counts):
        supported = exp in ALLOWED_EXPONENTS

        if supported:
            status = "SUPPORTED"
        else:
            status = "NOT SUPPORTED"

        print(
            f"2^{exp:3d} : "
            f"{count:8d} weights   [{status}]"
        )

    unsupported = [
        (int(exp), int(count))
        for exp, count in zip(unique_exp, counts)
        if int(exp) not in ALLOWED_EXPONENTS
    ]

    print("\n" + "-" * 60)
    print("Unsupported exponents:")
    print("-" * 60)

    if len(unsupported) == 0:
        print("NONE")
        print("All PoT weights are supported by current C++ code.")
    else:
        for exp, count in unsupported:
            print(
                f"2^{exp:3d} : "
                f"{count:8d} weights"
            )

    
    total = len(nonzero)

    unsupported_count = sum(
        count for _, count in unsupported
    )

    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)

    print(f"Non-zero weights       : {total}")
    print(f"Unsupported weights    : {unsupported_count}")

    if total > 0:
        percentage = (
            unsupported_count / total
        ) * 100.0

        print(
            f"Unsupported percentage : "
            f"{percentage:.4f}%"
        )

    print("=" * 60)


print("=" * 70)
print("MNIST Power-of-Two Approximation")
print("=" * 70)

print(f"Baseline model : {MODEL_PATH}")
print(f"Output model   : {OUTPUT_MODEL_PATH}")
print(f"MNIST dataset  : {MNIST_PATH}")
print(f"Threshold      : {SELECTED_THRESHOLD}")



print("\nLoading baseline model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
    metrics=["accuracy"]
)



print("\nLoading MNIST dataset...")

with np.load(MNIST_PATH) as data:
    x_test = data["x_test"].astype(np.float32)
    y_test = data["y_test"]

print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")



x_test = x_test / 255.0

print("\nEvaluating baseline model...")

baseline_loss, baseline_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(
    f"\nBaseline accuracy: "
    f"{baseline_accuracy * 100:.4f}%"
)




original_weights = model.get_weights()

print(
    f"\nNumber of weight tensors: "
    f"{len(original_weights)}"
)



approx_weights = []

total_weights = 0
total_changed = 0
total_pruned = 0


print("\n" + "=" * 70)
print("Applying Power-of-Two approximation")
print("=" * 70)


for i, weights in enumerate(original_weights):

    weights = np.asarray(
        weights,
        dtype=np.float32
    )

    total_weights += weights.size


    if weights.ndim >= 2:

        new_weights = apply_pot_to_array(
            weights,
            SELECTED_THRESHOLD
        )

        

        analyze_pot_exponents(
            new_weights,
            layer_name=f"Tensor {i}"
        )

    else:

       
        new_weights = weights.copy()

        print(
            f"\nTensor {i}: "
            f"bias/vector -> unchanged"
        )


    changed = np.sum(
        weights != new_weights
    )

    pruned = np.sum(
        (weights != 0) &
        (new_weights == 0)
    )

    total_changed += changed
    total_pruned += pruned

    approx_weights.append(
        new_weights.astype(np.float32)
    )



print("\n" + "=" * 70)
print("Global PoT statistics")
print("=" * 70)

print(
    f"Total parameters : "
    f"{total_weights}"
)

print(
    f"Changed values   : "
    f"{total_changed}"
)

print(
    f"Pruned values    : "
    f"{total_pruned}"
)

if total_weights > 0:

    print(
        f"Changed ratio    : "
        f"{100.0 * total_changed / total_weights:.4f}%"
    )

    print(
        f"Pruned ratio     : "
        f"{100.0 * total_pruned / total_weights:.4f}%"
    )


print("\nSetting approximate weights...")

model.set_weights(
    approx_weights
)

print("Approximate weights applied.")
print("\nEvaluating approximate PoT model...")

approx_loss, approx_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print(
    f"\nApproximate accuracy: "
    f"{approx_accuracy * 100:.4f}%"
)

print("\nSaving approximate model...")

model.save(
    OUTPUT_MODEL_PATH
)

print(
    f"Saved successfully to:\n"
    f"{OUTPUT_MODEL_PATH}"
)

