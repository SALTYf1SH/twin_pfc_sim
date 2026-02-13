
import glob
import os
import numpy as np

DATASET_DIR = os.path.join(os.getcwd(), "..", "final_dataset_stress")

def get_file_hash(filenames):
    return hash(tuple(filenames))

print(f"Checking glob stability in: {DATASET_DIR}")

# Run 1
files1 = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
print(f"Run 1: {len(files1)} files. First 3: {[os.path.basename(f) for f in files1[:3]]}")

# Run 2
files2 = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
print(f"Run 2: {len(files2)} files. First 3: {[os.path.basename(f) for f in files2[:3]]}")

if files1 == files2:
    print("Glob order appears stable between immediate calls.")
else:
    print("Glob order is UNSTABLE!")

# Check Sorted Shuffle
np.random.seed(42)
sorted_files = sorted(files1)
shuffled_sorted_1 = np.copy(sorted_files)
np.random.shuffle(shuffled_sorted_1)

np.random.seed(42)
shuffled_unsorted_1 = np.copy(files1)
np.random.shuffle(shuffled_unsorted_1)

if list(shuffled_sorted_1) == list(shuffled_unsorted_1):
    print("Original glob was already sorted.")
else:
    print("Original glob was NOT sorted. Sorting changes the shuffle result.")
