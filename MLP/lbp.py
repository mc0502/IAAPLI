import cv2
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern

# Function to calculate LBP histogram for an image
def calculate_lbp_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(256 + 1))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    return hist


# Root directory of the dataset
dataset_root = './../melanoma_cancer_dataset'

# List of classes
classes = ['benign', 'malignant']

for phase in ['test', 'train']:
    phase_path = os.path.join(dataset_root, phase)
    print(phase_path)
    
    # Create a list to store the LBP features for the current phase
    lbp_features = []
    
    for class_label in classes:
        class_path = os.path.join(phase_path, class_label)
        print(class_path)
        
        # List all the image files in the current class directory
        image_files = os.listdir(class_path)
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            
            image = cv2.imread(image_path, 1)
            lbp_hist = calculate_lbp_histogram(image)

            # Extend the class_label_list with lbp_hist values
            lbp_features.append([class_label] + lbp_hist.tolist())
            print(image_path)

    # Save the LBP features to a CSV file for the current phase and class label
    csv_file = f'lbp_features_{phase}.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lbp_features)

    print("LBP features for", phase, "are saved in", csv_file)

