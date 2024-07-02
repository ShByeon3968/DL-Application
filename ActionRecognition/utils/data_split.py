import os
import random
import shutil

def split_files(source_folder1, source_folder2, train_folder, test_folder, train_txt, test_txt, split_ratio=0.7):
    # Create train and test folders if they do not exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Get list of mp4 files from both source folders
    files1 = [os.path.join(source_folder1, f) for f in os.listdir(source_folder1) if f.endswith('.mp4') and os.path.isfile(os.path.join(source_folder1, f))]
    files2 = [os.path.join(source_folder2, f) for f in os.listdir(source_folder2) if f.endswith('.mp4') and os.path.isfile(os.path.join(source_folder2, f))]

    # Combine files and shuffle
    files = [(f, 1) for f in files1] + [(f, 2) for f in files2]
    random.shuffle(files)

    # Split files into train and test sets
    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    test_files = files[split_point:]

    # Write train and test file paths to their respective txt files
    with open(train_txt, 'w') as train_f:
        for f, folder_index in train_files:
            destination = os.path.join(train_folder, os.path.basename(f))
            shutil.copy2(f, destination)
            train_f.write(f"{destination} {folder_index}\n")

    with open(test_txt, 'w') as test_f:
        for f, folder_index in test_files:
            destination = os.path.join(test_folder, os.path.basename(f))
            shutil.copy2(f, destination)
            test_f.write(f"{destination} {folder_index}\n")

# Define your folders
source_folder1 = 'path/to/source_folder1'
source_folder2 = 'path/to/source_folder2'
train_folder = 'path/to/train_folder'
test_folder = 'path/to/test_folder'
train_txt = 'path/to/train.txt'
test_txt = 'path/to/test.txt'

# Run the function
split_files(source_folder1, source_folder2, train_folder, test_folder, train_txt, test_txt)
