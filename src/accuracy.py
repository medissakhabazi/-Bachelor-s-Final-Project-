import os
import numpy as np
from sklearn.metrics import precision_score, recall_score

N = 10000


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MNIST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mnist.npz"
)

CSIM_FILE_exact = "result/csim_results_exact.log"
CSIM_FILE_approximate = "result/csim_results_approximate.log"


print("*" * 50)
print("Loading MNIST labels")
print("*" * 50)

if not os.path.exists(MNIST_PATH):
    raise FileNotFoundError(f"MNIST dataset not found: {MNIST_PATH}")

with np.load(MNIST_PATH) as data:
    y_test = data["y_test"]

y = y_test[:N]

print("Number of labels:", len(y))


print()
print("*" * 50)
print("Loading C-Simulation results")
print("*" * 50)

if not os.path.exists(CSIM_FILE_exact):
    raise FileNotFoundError(f"C-SIM_exact result file not found: {CSIM_FILE_exact}")

results_exact = np.loadtxt(CSIM_FILE_exact)


if not os.path.exists(CSIM_FILE_approximate):
    raise FileNotFoundError(f"C-SIM_approximate result file not found: {CSIM_FILE_approximate}")

results_approximate = np.loadtxt(CSIM_FILE_approximate)

print("C-SIM result shape:", results_exact.shape)

if results_exact.shape[0] != N:
    raise ValueError(
        f"Expected {N} samples, "
        f"but C-SIM_exact produced {results_exact.shape[0]} samples."
    )

if results_exact.shape[1] != 10:
    raise ValueError(
        f"Expected 10 outputs per sample, "
        f"but got {results_exact.shape[1]}."
    )



if results_approximate.shape[0] != N:
    raise ValueError(
        f"Expected {N} samples, "
        f"but C-SIM_approximate produced {results_approximate.shape[0]} samples."
    )

if results_approximate.shape[1] != 10:
    raise ValueError(
        f"Expected 10 outputs per sample, "
        f"but got {results_approximate.shape[1]}."
    )


exact_predicted = np.argmax(results_exact, axis=1)
approximate_predicted = np.argmax(results_approximate , axis=1)


exact_correct = np.sum(exact_predicted == y)
total = len(y)

exact_accuracy = exact_correct / total


exact_precision = precision_score(
    y,
    exact_predicted,
    average="macro",
    zero_division=0
)

exact_recall = recall_score(
    y,
    exact_predicted,
    average="macro",
    zero_division=0
)


approximate_correct = np.sum(approximate_predicted == y)

approximate_accuracy = approximate_correct / total


approximate_precision = precision_score(
    y,
    approximate_predicted,
    average="macro",
    zero_division=0
)

approximate_recall = recall_score(
    y,
    approximate_predicted,
    average="macro",
    zero_division=0
)



print()
print("*" * 50)
print("exact:")
print("*" * 50)
print()
print(f"Exact_Correct predictions : {exact_correct}")
print(f"Exact_Wrong predictions   : {total - exact_correct}")
print(f"Total samples       : {total}")
print(f"exact_Accuracy             : {exact_accuracy * 100:.4f}%")
print(f"exact_Precision : {exact_precision * 100:.4f}%")
print(f"exact_Recall    : {exact_recall * 100:.4f}%")

print()
print("*" * 50)
print("approximate:")
print("*" * 50)
print()
print(f"approximate_Correct predictions : {approximate_correct}")
print(f"approximate_Wrong predictions   : {total - approximate_correct}")
print(f"Total samples       : {total}")
print(f"approximate_Accuracy             : {approximate_accuracy * 100:.4f}%")
print(f"approximate_Precision : {approximate_precision * 100:.4f}%")
print(f"approximate_Recall    : {approximate_recall * 100:.4f}%")

