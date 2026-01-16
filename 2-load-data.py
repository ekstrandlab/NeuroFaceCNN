import os
import numpy as np
import tensorflow as tf 
import pickle
import logging
from pathlib import Path

#logging set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PROJECT_ROOT = Path(__file__).resolve().parent

MATRICES_DIR = PROJECT_ROOT / "Output" / "2D-matrices"
FACE_DIR = MATRICES_DIR / "face"
NOFACE_DIR = MATRICES_DIR / "noface"


#counting the number of .txt files in all subject subfolders
def count_text_files(folder_name):
    folder_path = os.path.join(BASE_PATH, folder_name)
    count = sum(
        file.endswith('.txt')
        for subject in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, subject))
        for file in os.listdir(os.path.join(folder_path, subject))
    )
    return count


#load .txt data from a specified folder and return as a list of 2D NumPy arrays
def load_text_data(folder_name, limit=None):
    
    data = []
    subject_path = os.path.join(BASE_PATH, folder_name)
    subject_folders = sorted(os.listdir(subject_path))
    for idx, subject in enumerate(subject_folders):
        if limit is not None and idx >= limit:
            break
        full_subject_path = os.path.join(subject_path, subject)
        if os.path.isdir(full_subject_path):
            for file in sorted(os.listdir(full_subject_path)):
                if file.endswith(".txt"):
                    file_path = os.path.join(full_subject_path, file)
                    matrix = np.loadtxt(file_path, delimiter=",")
                    data.append(matrix)
            logging.info(f"Loaded subject: {subject} ({folder_name})")
    logging.info(f"Processed {min(len(subject_folders), limit or len(subject_folders))} subjects from '{folder_name}'.")
    return data


def main():
    logging.info("Counting input files...")
    face_count = count_npy_files(FACE_DIR)
    noface_count = count_npy_files(NOFACE_DIR)
    logging.info(f"Found {face_count} '.npy' files in '{FACE_DIR}'")
    logging.info(f"Found {noface_count} '.npy' files in '{NOFACE_DIR}'")

    logging.info("Loading data...")
    face_data = load_npy_data(FACE_DIR)
    noface_data = load_npy_data(NOFACE_DIR)
    
    logging.info("Constructing input arrays...")
    X = np.array(face_data + noface_data)
    #label
    y = np.array([1] * len(face_data) + [0] * len(noface_data))

    #reshape for CNN input: (samples, height, width, channels)
    height, width = X.shape[1], X.shape[2]
    X = X.reshape(-1, height, width, 1)

    logging.info(f"X shape: {X.shape}, y shape: {y.shape}, X size: {X.nbytes / 1e6:.2f} MB")


    output_path = MATRICES_DIR / "fmri_data.pickle"
    logging.info(f"Saving preprocessed data to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    logging.info("Data successfully saved.")


if __name__ == "__main__":
    main()

#3606 files, 41489 rows, 10 timepoint, 1 channel 
#face: 1 / noface: 0
