import os
import shutil
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# Load the trained YOLOv8 model
train_dir = 'runs/detect'
train_folders = [f for f in os.listdir(train_dir) if f.startswith('train') and os.path.isdir(os.path.join(train_dir, f))]

def folder_key(x): 
    if x == "train":
        return 1
    return int(x.replace("train", " "))

train_folders_sorted = sorted(train_folders, key=folder_key)
latest_train_folder = train_folders_sorted[-1]

latest_train_path = os.path.join(train_dir, latest_train_folder)

weights_folder = os.path.join(latest_train_path, "weights")

model = YOLO(os.path.join(weights_folder, "best.pt"))
print(type(model))

# Define the source folder and target folder
source_folder = "test_run/images"
defects_folder = "test_run/defects"

label_thresholds = {
    'lviv_open': 0.9,
    'wfl_open': 0.9
}

# Check if defects folder exists
if os.path.exists(defects_folder):
    # If the folder exists, delete all contents
    for filename in os.listdir(defects_folder):
        file_path = os.path.join(defects_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the folder
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
else:
    # Create the defects folder if it doesn't exist
    os.makedirs(defects_folder)

poses_csv_path = os.path.join(source_folder, "poses.csv")
poses_df = pd.read_csv(poses_csv_path, header=None) 
count = 0
curr_location = None
curr_defect = None
prev_defect = None

# Loop through the images in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
        img_path = os.path.join(source_folder, filename)

        # Check if the image can be opened
        try:
            img = Image.open(img_path)
            img.verify()  # Verify the image is valid
        except (IOError, SyntaxError) as e:
            print(f"Image {filename} is invalid. Error: {e}")
            continue  # Skip to the next image

        # Run YOLO detection
        results = model(img_path)

        valid_detection = False
        detected_label = None

        for detection in results[0].boxes:
            label = detection.cls  
            confidence = detection.conf  
            label_name = model.names[int(label)]
            
            if label_name in label_thresholds: 
            #and confidence >= label_thresholds[label_name]:
                valid_detection = True
                detected_label = label_name
                break  

        if valid_detection and detected_label:
            # Plot the detections on the image
            result_img = results[0].plot()  # This returns an image array with the boxes drawn
            result_pil_image = Image.fromarray(np.uint8(result_img[:,:,[2,1,0]]))
    
            # Save the image with bounding boxes to the defects folder
            filename_without_extension = os.path.splitext(filename)[0]
            print(f"file name without extension is {filename_without_extension}")
            row_index = int(filename_without_extension) + 1
            print("row index")
            value_from_csv = poses_df.iloc[row_index, 0] 
    
            print(f"Value from CSV for image {filename_without_extension}: {value_from_csv}")

            if -80 <= value_from_csv < -55:
                carriage_num = "DT2"
            elif -55 <= value_from_csv < -32:
                carriage_num = "MP2"
            elif -32 <= value_from_csv < -10:
                carriage_num = "MI2" 
            elif -10 <= value_from_csv < 12:
                carriage_num = "MI1" 
            elif 12 <= value_from_csv < 35:
                carriage_num = "MP1" 
            elif 35 <= value_from_csv < 60:
                carriage_num = "DT1"  
            
            # count += 1 
            # curr_defect = detected_label
            curr_location = carriage_num

            # if (count == 55) and (curr_defect != prev_defect):  
            #     prev_defect = curr_defect

            new_filename = f"{detected_label}_{filename_without_extension}_{curr_location}.jpg"
            output_path = os.path.join(defects_folder, new_filename)
            result_pil_image.save(output_path)

print("finished all files")

            # elif count>55: 
            #     if curr_defect != prev_defect: 
            #         count = 0
            #     else: 
            #         print("not writing file")
                