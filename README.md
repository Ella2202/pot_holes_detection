# YOLOv5 Pothole Detection – README
## Project Overview
This project trains a **YOLOv5 model** to detect potholes using a custom dataset. The workflow includes:
* Downloading dataset from Kaggle
* Parsing XML annotations (Pascal VOC format)
* Converting annotations to YOLOv5 format
* Visualizing bounding boxes
* Train/Val split creation
* Data augmentation using **Albumentations**
* Training YOLOv5s model on the prepared dataset
## Folder Structure
After running all preprocessing steps, your dataset structure becomes:

```
archive/
 ├── images/
 │   ├── train/
 │   └── val/
 ├── labels/
 │   ├── train/
 │   └── val/
 ├── annotations/   # converted YOLO txt files
```

## Step-by-Step Pipeline

### Clone YOLOv5 and Install Dependencies

```
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt
%pip install -q roboflow
```

The script confirms PyTorch + GPU availability.

### Load Dataset from Kaggle

```
path = kagglehub.dataset_download("farzadnekouei/pothole-image-segmentation-dataset")
```

The dataset includes:

* **Images**
* **XML annotations** (Pascal VOC format)

### XML Parsing → Extracting Bounding Boxes

The script extracts:

* filename
* image size
* bounding box coordinates
* class name

Using:

```python
def extract_info_from_xml(xml_file):
    ...
```

### Converting XML → YOLOv5 TXT Format

YOLOv5 format requires:

```
class x_center y_center width height  (normalized)
```

Conversion code:

```python
def convert_to_yolov5(info_dict):
    ...
```

TXT files are stored in:

```
/content/archive/annotations
```

### Visualizing Bounding Boxes

A helper function overlays YOLO boxes on the image using Pillow:

```python
def plot_bounding_box(image, annotation_list):
    ...
```

---

### Organizing Train/Validation Splits

```
train_test_split(..., test_size=0.2)
```

Files are copied into the YOLO directory structure.

## Data Augmentation

Using **Albumentations**:

* RandomCrop
* HorizontalFlip
* RandomBrightnessContrast
* ShiftScaleRotate

Transform definition:

```python
transform = A.Compose([...], bbox_params=A.BboxParams(format='yolo'))
```

Augmented images + labels are saved back into train/val folders.

## Training YOLOv5

The dataset config file `potholes.yaml` contains:

```
path: /content/archive/
train: images/train
val: images/val
nc: 1
names: ['pothole']
```

Train command:

```
!python train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch.yaml \
 --batch 32 --epochs 16 --data potholes.yaml --weights yolov5s.pt
```

The training script logs:

* Precision (P)
* Recall (R)
* mAP@0.5
* mAP@0.5:0.95

You can visualize metrics in:

```
runs/train/expX/
```

## Inference After Training

Example inference:

```
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --source your_image.jpg
```

Results appear in:

```
runs/detect/exp/
```


