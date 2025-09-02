import os
import yaml
from PIL import Image
import shutil
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error, accuracy_score
import gc
from torch import cuda
from IPython.display import display

"""# Dataset Preparation

All the data was extracted from its current structure, processed, and reorganized in a way suitable for training the model under the folder named "dataset." The annotation files were recalculated and reformatted according to the format used by YOLO models. Additionally, a data.yaml file was created.
"""

cwd = os.getcwd()
dataset_path = os.path.join(cwd, 'dataset')
car_dataset_path = os.path.join(cwd, 'cars_dataset')
print(dataset_path)

os.mkdir(dataset_path)

# Paths
image_dir = os.path.join(car_dataset_path, 'Images')
label_input_dir = os.path.join(car_dataset_path, 'Annotations')
image_set_dir = os.path.join(car_dataset_path, 'ImageSets')

label_output_dir = os.path.join(dataset_path, 'labels')
image_output_dir = os.path.join(dataset_path, 'images')

os.mkdir(label_output_dir)
os.mkdir(image_output_dir)

for filename in os.listdir(image_set_dir):
    set_path = os.path.join(image_set_dir, filename)
    label_output_path = os.path.join(label_output_dir, filename.replace(".txt", ""))
    os.mkdir(label_output_path)
    image_output_path = os.path.join(image_output_dir, filename.replace(".txt", ""))
    os.mkdir(image_output_path)

    with open(set_path, 'r') as f:
        for line in f:
            image_name = "".join([line.strip(), ".png"])
            source = os.path.join(image_dir, image_name)
            destination = os.path.join(image_output_path, image_name)
            dest = shutil.copyfile(source, destination)

            annotation_name = "".join([line.strip(), ".txt"])
            label_path = os.path.join(label_input_dir, annotation_name)

            # Get image size
            with Image.open(destination) as img:
                img_width, img_height = img.size

            yolo_lines = []
            with open(label_path, 'r') as f2:
                for line2 in f2:
                    parts = line2.strip().split()
                    class_id = 0
                    x_min = float(parts[0])
                    y_min = float(parts[1])
                    x_max = float(parts[2])
                    y_max = float(parts[3])

                    # Convert to YOLO format
                    x_center = (x_min + x_max) / 2.0 / img_width
                    y_center = (y_min + y_max) / 2.0 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)

            # Save new label file
            with open(os.path.join(label_output_path, annotation_name), 'w') as f:
                f.write('\n'.join(yolo_lines))

data = {
'path': dataset_path,  # dataset root directory
'train': 'images/train',  # train images folder
'val': 'images/val',  # validation images folder
'test': 'images/test',  # test images folder

'nc': 1,
'names': ["car"]
}

with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
    yaml.dump(data, f, sort_keys=False)

"""# Model Training and Hyperparameter Tuning

Functions used to easily experiment with the model using different hyperparameters and to conveniently print the results.
"""

def print_plots(name):
    runs_dir = os.path.join(cwd, 'runs\detect')
    res_dir = os.path.join(runs_dir, name)
    result_png = os.path.join(res_dir, 'results.png')
    img = Image.open(result_png)
    display(img)

def tune_model(batch, lr, freeze, optimizer='AdamW', box=7.5, cls=0.5, dfl=1.5, iou=0.7, del_model=True, print_plot=True):
    # Load a COCO-pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    name = "freeze=" + str(freeze) + ", optimizer=" + optimizer + ", batch=" + str(batch) + ", lr=" + str(lr) + ", box=" + str(box) + ", cls=" + str(cls) + ", dfl=" + str(dfl) + ", iou=" + str(iou)

    _ = model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=40, imgsz=640, batch=batch, lr0=lr, optimizer=optimizer, patience=10, freeze=freeze, box=box, cls=cls, dfl=dfl, iou=iou, name=name)

    if print_plot:
        print_plots(name)

    if del_model:
        del model
        gc.collect()
        cuda.empty_cache()
    else:
        return model

"""Since all four different training setups were optimized in the same way during hyperparameter tuning, they will be interpreted collectively.

## Train the entire model:

### Learning Hyperparameters:

#### Batch = 8, LR = 0.004, SGD Optimizer:
"""

tune_model(batch=8, lr=0.004, freeze=0, optimizer='SGD')

"""#### Batch = 8, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=8, lr=0.004, freeze=0, optimizer='AdamW')

"""#### Batch = 8, LR = 0.002, AdamW Optimizer:"""

tune_model(batch=8, lr=0.002, freeze=0, optimizer='AdamW')

"""#### Batch = 8, LR = 0.001, AdamW Optimizer:"""

tune_model(batch=8, lr=0.001, freeze=0, optimizer='AdamW')

"""#### Batch = 4, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=4, lr=0.004, freeze=0, optimizer='AdamW')

"""### YOLO Specific Hyperparameters:

#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 1.5, IOU Threshold = 0.7:
"""

tune_model(batch=8, lr=0.004, freeze=0, optimizer='AdamW', box=5, cls=0.5, dfl=1.5, iou=0.7)

"""#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=0, optimizer='AdamW', box=5, cls=0.5, dfl=2, iou=0.5)

"""#### Box Loss = 5, Class Loss = 0.7, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=0, optimizer='AdamW', box=5, cls=0.7, dfl=2, iou=0.5)

"""## Freeze the first 5 blocks:

### Learning Hyperparameters:

#### Batch = 8, LR = 0.004, SGD Optimizer:
"""

tune_model(batch=8, lr=0.004, freeze=5, optimizer='SGD')

"""#### Batch = 8, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=8, lr=0.004, freeze=5, optimizer='AdamW')

"""#### Batch = 8, LR = 0.002, AdamW Optimizer:"""

tune_model(batch=8, lr=0.002, freeze=5, optimizer='AdamW')

"""#### Batch = 8, LR = 0.001, AdamW Optimizer:"""

tune_model(batch=8, lr=0.001, freeze=5, optimizer='AdamW')

"""#### Batch = 4, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=4, lr=0.004, freeze=5, optimizer='AdamW')

"""### YOLO Specific Hyperparameters:

#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 1.5, IOU Threshold = 0.7:
"""

tune_model(batch=8, lr=0.004, freeze=5, optimizer='AdamW', box=5, cls=0.5, dfl=1.5, iou=0.7)

"""#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=5, optimizer='AdamW', box=5, cls=0.5, dfl=2, iou=0.5)

"""#### Box Loss = 5, Class Loss = 0.7, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=5, optimizer='AdamW', box=5, cls=0.7, dfl=2, iou=0.5)

"""## Freeze the first 10 blocks:

### Learning Hyperparameters:

#### Batch = 8, LR = 0.004, SGD Optimizer:
"""

tune_model(batch=8, lr=0.004, freeze=10, optimizer='SGD')

"""#### Batch = 8, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=8, lr=0.004, freeze=10, optimizer='AdamW')

"""#### Batch = 8, LR = 0.002, AdamW Optimizer:"""

tune_model(batch=8, lr=0.002, freeze=10, optimizer='AdamW')

"""#### Batch = 8, LR = 0.001, AdamW Optimizer:"""

tune_model(batch=8, lr=0.001, freeze=10, optimizer='AdamW')

"""#### Batch = 4, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=4, lr=0.004, freeze=10, optimizer='AdamW')

"""### YOLO Specific Hyperparameters:

#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 1.5, IOU Threshold = 0.7:
"""

tune_model(batch=8, lr=0.004, freeze=10, optimizer='AdamW', box=5, cls=0.5, dfl=1.5, iou=0.7)

"""#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=10, optimizer='AdamW', box=5, cls=0.5, dfl=2, iou=0.5)

"""#### Box Loss = 5, Class Loss = 0.7, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=10, optimizer='AdamW', box=5, cls=0.7, dfl=2, iou=0.5)

"""## Freeze the first 21 blocks:

### Learning Hyperparameters:

#### Batch = 8, LR = 0.004, SGD Optimizer:
"""

tune_model(batch=8, lr=0.004, freeze=21, optimizer='SGD')

"""#### Batch = 8, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=8, lr=0.004, freeze=21, optimizer='AdamW')

"""#### Batch = 8, LR = 0.002, AdamW Optimizer:"""

tune_model(batch=8, lr=0.002, freeze=21, optimizer='AdamW')

"""#### Batch = 8, LR = 0.001, AdamW Optimizer:"""

tune_model(batch=8, lr=0.001, freeze=21, optimizer='AdamW')

"""#### Batch = 4, LR = 0.004, AdamW Optimizer:"""

tune_model(batch=4, lr=0.004, freeze=21, optimizer='AdamW')

"""### YOLO Specific Hyperparameters:

#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 1.5, IOU Threshold = 0.7:
"""

tune_model(batch=8, lr=0.004, freeze=21, optimizer='AdamW', box=5, cls=0.5, dfl=1.5, iou=0.7)

"""#### Box Loss = 5, Class Loss = 0.5, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=21, optimizer='AdamW', box=5, cls=0.5, dfl=2, iou=0.5)

"""#### Box Loss = 5, Class Loss = 0.7, DFL Loss = 2, IOU Threshold = 0.5:"""

tune_model(batch=8, lr=0.004, freeze=21, optimizer='AdamW', box=5, cls=0.7, dfl=2, iou=0.5)

"""## Discussion

For the four different freezing setups, both learning and YOLO-specific parameters were tested in various combinations. Since all four models showed improvements when similar configurations were applied, they were evaluated collectively. Initially, learning parameters were explored.

The first learning parameter tested was the optimization algorithm. The classic SGD algorithm was compared with the more modern AdamW algorithm. AdamW provided clear performance improvements in all scenarios, thanks to its ability to adjust learning rates per parameter and the regularization it offers internally. Afterwards, different learning rates were tested. Despite reducing the existing learning rate, no improvement in learning performance was observed, and the model remained underfit after 40 epochs. Thus, the initial value of 0.004 was chosen as optimal. Later, batch size was modified from the memory-optimized batch=8 to batch=4, but this also did not improve performance.

Next, YOLO-specific parameters were tested while keeping the previously optimized parameters fixed. The default box loss weight of 7 was considered high, and a lower value of 5 was tested to relatively increase the importance of classification. This change yielded a slight performance improvement. Subsequently, the DFL loss weight was increased to enhance bounding box accuracy, and the IoU threshold was lowered to reduce the model's oversensitivity. Again, small improvements were observed in overall metrics across all models. Finally, the class loss weight was increased further to boost classification performance, which also led to improved results.

In conclusion, the best-performing configurations were those tested last, and these were used for the remainder of the experiments.

# Visualized Bounding Box Predictions
"""

def print_bounding_boxes(freeze, number, label=False):
    name = "freeze=" + str(freeze) + ", optimizer=AdamW, batch=8, lr=0.004, box=5, cls=0.7, dfl=2, iou=0.5"
    runs_dir = os.path.join(cwd, 'runs\detect')
    res_dir = os.path.join(runs_dir, name)

    if label:
        extension = "labels"
    else:
        extension = "pred"

    box_jpg = os.path.join(res_dir, "val_batch" + str(number) + "_" + extension + ".jpg")
    img = Image.open(box_jpg)
    display(img)

"""Below are examples of the bounding boxes generated by each model on the validation dataset, along with the corresponding ground truth bounding boxes.

## Freeze = 0 Model:
"""

print_bounding_boxes(0, 0, label=False)

print_bounding_boxes(0, 0, label=True)

print_bounding_boxes(0, 1, label=False)

print_bounding_boxes(0, 1, label=True)

"""## Freeze = 5 Model:"""

print_bounding_boxes(5, 0, label=False)

print_bounding_boxes(5, 0, label=True)

print_bounding_boxes(5, 1, label=False)

print_bounding_boxes(5, 1, label=True)

"""## Freeze = 10 Model:"""

print_bounding_boxes(10, 0, label=False)

print_bounding_boxes(10, 0, label=True)

print_bounding_boxes(10, 1, label=False)

print_bounding_boxes(10, 1, label=True)

"""## Freeze = 21 Model:"""

print_bounding_boxes(21, 0, label=False)

print_bounding_boxes(21, 0, label=True)

print_bounding_boxes(21, 1, label=False)

print_bounding_boxes(21, 1, label=True)

"""# Metric Calculations

Function used to test the models in a practical way and to measure their performance metrics.
"""

runs_dir = os.path.join(cwd, 'runs\detect')

def test_model(model):
    test_labels_path = os.path.join(dataset_path, 'labels\\test')
    test_images_path = os.path.join(dataset_path, 'images\\test')

    real_counts = []
    for filename in os.listdir(test_labels_path):
        with open(os.path.join(test_labels_path, filename), 'r') as f:
            real_counts.append(len(f.readlines()))

    pred_results = model(test_images_path)

    pred_counts = []
    for res in pred_results:
        pred_counts.append(len(res.boxes))

    exact_match_acc = accuracy_score(real_counts, pred_counts) * 100
    mse = mean_squared_error(real_counts, pred_counts)

    return exact_match_acc, mse

"""## Freeze = 0 Model:"""

name = "freeze=0, optimizer=AdamW, batch=8, lr=0.004, box=5, cls=0.7, dfl=2, iou=0.5"
model = YOLO(os.path.join(runs_dir, name + "\\weights\\best.pt"))

exact_match_acc, mse = test_model(model)

print("Metrics for the Freeze 0 Model:\n")
print("Exact Match Accuracy: " + str(exact_match_acc))
print("Mean Squared Error: " + str(mse))

"""## Freeze = 5 Model:"""

name = "freeze=5, optimizer=AdamW, batch=8, lr=0.004, box=5, cls=0.7, dfl=2, iou=0.5"
model = YOLO(os.path.join(runs_dir, name + "\\weights\\best.pt"))

exact_match_acc, mse = test_model(model)

print("Metrics for the Freeze 5 Model:\n")
print("Exact Match Accuracy: " + str(exact_match_acc))
print("Mean Squared Error: " + str(mse))

"""## Freeze = 10 Model:"""

name = "freeze=10, optimizer=AdamW, batch=8, lr=0.004, box=5, cls=0.7, dfl=2, iou=0.5"
model = YOLO(os.path.join(runs_dir, name + "\\weights\\best.pt"))

exact_match_acc, mse = test_model(model)

print("Metrics for the Freeze 10 Model:\n")
print("Exact Match Accuracy: " + str(exact_match_acc))
print("Mean Squared Error: " + str(mse))

"""## Freeze = 21 Model:"""

name = "freeze=21, optimizer=AdamW, batch=8, lr=0.004, box=5, cls=0.7, dfl=2, iou=0.5"
model = YOLO(os.path.join(runs_dir, name + "\\weights\\best.pt"))

exact_match_acc, mse = test_model(model)

print("Metrics for the Freeze 21 Model:\n")
print("Exact Match Accuracy: " + str(exact_match_acc))
print("Mean Squared Error: " + str(mse))

"""## Discussion

When examining the results, it can be observed that the MSE value improves as the freeze level decreases. Exact match accuracy is slightly better for freeze=5 and freeze=10 compared to freeze=0, while it is clearly the worst at freeze=21. As seen in this case, exact match accuracy can sometimes produce inconsistent results due to chance, especially when model performances are relatively close. Considering that the MSE metric accounts for the magnitude of poor predictions in each image and increases exponentially with higher deviation, it may be regarded as a more reliable evaluation criterion. In this context, as expected, the model's ability to adapt to the dataset decreases as the number of frozen layers increases. Nevertheless, depending on the dataset, freeze values of 5 and 10 could still be acceptable if slight performance losses are tolerable in exchange for reduced computational cost.

To further improve performance:

* A larger and more recent YOLO model could be used.
* The dataset could be expanded.
* The input image resolution could be increased.
* Models specifically trained for counting tasks could be employed.

# Alternative Methods

In addition to using object detection models like YOLO, there are several other methods commonly used for solving object counting problems in computer vision. These methods vary in complexity and are suited to different types of scenarios. Below are brief descriptions of some alternative approaches:

1. Density Map Estimation
This approach involves training a model to produce a density map of an image, where the sum of all pixel values in the map corresponds to the total object count. This method is particularly effective in crowded scenes where individual objects overlap and are difficult to detect separately. Models such as CSRNet and MCNN are examples of networks used for this purpose.

2. Regression-Based Counting
In regression-based methods, a convolutional neural network (CNN) is trained to directly predict the number of objects in an image as a single scalar value. This technique does not provide localization but can be efficient and accurate in cases where spatial information is not critical.

3. Segmentation-Based Methods
These methods involve segmenting each object instance in an image using techniques like instance or semantic segmentation. The number of segmented instances is then counted. A popular model used for this approach is Mask R-CNN. This method is useful when both object count and shape or area information are required.

4. Feature Extraction and Clustering
In this traditional computer vision approach, object-like features (e.g., blobs, corners) are first extracted from the image. These features are then grouped using clustering algorithms such as K-means or DBSCAN. The number of clusters can give an estimate of the object count. This method can be effective in simpler settings or when limited computational resources are available.

5. Transformer-Based Models
Vision transformers can also be adapted for counting tasks by treating the image as a sequence of patches and learning contextual relationships between them. These models can be trained to predict object counts directly and are becoming more popular due to their high performance on complex vision tasks.
"""