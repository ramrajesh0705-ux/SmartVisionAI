# SmartVision AI

A computer-vision project that compares transfer-learned image classifiers with a fine-tuned YOLOv8 detector and exposes both workflows through Streamlit.

🚀 **[Live Demo →](https://huggingface.co/spaces/rajesh7593/SmartVisionAI)**

## Problem statement

Recognize objects from a COCO-derived subset in two different ways: classify a single cropped/object image, or detect multiple objects with bounding boxes in a full image.

## Key objectives

- Build a reproducible image dataset and train/compare four pretrained CNN backbones.
- Fine-tune YOLOv8 for multi-object detection.
- Evaluate classification and detection with task-appropriate metrics.
- Provide an interactive inference demo for both tasks.

## Technologies

- Python, PyTorch, Torchvision
- Ultralytics YOLOv8
- Streamlit, OpenCV, Pillow
- NumPy, pandas, scikit-learn, Plotly, Matplotlib

## Dataset and classes

The notebooks stream COCO data, select object categories, crop objects for classification, and create YOLO-format images and labels for detection. The preparation notebook records **2,500 classification images** (100 per selected category), split 70/15/15 into train/validation/test, and **2,125 detection images** with **10,986 annotated objects**.

> **Repository note:** the preparation notebook describes the selection as 25 classes. The configured class list is: person, bicycle, car, motorcycle, airplane, bus, train, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, couch, potted plant, and bed.

## Workflow

1. Prepare COCO-derived classification crops and YOLO annotations in `src/Smartvision.ipynb`.
2. Train four image classifiers from ImageFolder train/validation splits.
3. Evaluate classifiers on the test split and compare saved training histories.
4. Fine-tune pretrained `yolov8n.pt` for 50 epochs at 640px input resolution.
5. Validate YOLOv8 and use the saved weights for Streamlit inference.

## Image classification

Implemented in PyTorch/Torchvision with ImageNet-pretrained:

- VGG16
- ResNet50
- MobileNetV2
- EfficientNet-B0

The training scripts replace the ImageNet classifier head with a project-specific output head. VGG16 and MobileNetV2 freeze the convolutional feature extractor; ResNet50 freezes most parameters and unfreezes the final parameter set; EfficientNet-B0 uses differential learning rates, augmentation, early stopping, and mixed precision. Training histories are saved as JSON, and `evaluate_classification.py` reports test accuracy and a per-class classification report.

## Object detection

`src/train_yolo.py` fine-tunes pretrained YOLOv8 Nano using the dataset configuration in `data/detection/data.yaml`. `src/evaluate_yolo.py` validates the trained detector and reports mAP@0.5 and mAP@0.5:0.95. The Streamlit demo renders confidence-filtered bounding boxes and inference time.

## Results currently supported by the repository

- **YOLOv8:** the committed `models/yolo_finetuned/results.csv` reaches **mAP@0.5 = 0.8393** and **mAP@0.5:0.95 = 0.4872** at epoch 50.
- **Classification:** per-model JSON histories and a test-set evaluator are committed, but no consolidated numeric classification results are published in the repository README. The performance table in the Streamlit page is explicitly marked as example data and should not be treated as an experiment result.

## Application / demo

`app/app.py` implements a Streamlit interface with:

- **Image Classification:** upload an image and compare top-5 predictions from all four classifiers.
- **Object Detection:** upload an image, adjust the confidence threshold, and view YOLOv8 boxes, labels, scores, and measured inference time.
- **Model Performance:** a Plotly comparison page (currently uses example metrics).

Run the app only after supplying the expected trained weights under `models/` and the dataset/artifacts required by the loaders.

## Project structure

```text
app/app.py                         Streamlit application
src/Smartvision.ipynb              COCO subset preparation and dataset generation
src/train_*_classification.py      Classifier training scripts
src/evaluate_classification.py     Test accuracy and classification reports
src/model_compare.py               Training-history comparison and plots
src/train_yolo.py                  YOLOv8 fine-tuning
src/evaluate_yolo.py               YOLO validation metrics
src/config.py, src/utils.py         Paths, classes, model and inference helpers
data/detection/data.yaml           YOLO dataset configuration
models/*_history.json              Saved classifier training histories
models/yolo_finetuned/              YOLO run configuration, results, and plots
requirements.txt                    Python dependencies
```

## How to run

```bash
git clone https://github.com/ramrajesh0705-ux/SmartVisionAI.git
cd SmartVisionAI
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Place the generated dataset at `data/classification/{train,val,test}` and `data/detection/`, then train or provide the expected model files in `models/`:

```bash
python src/train_mobilenet_classification.py
python src/train_ResNet50_classification.py
python src/train_efficientnet_classification.py
python src/train_vgg16classification.py
python src/evaluate_classification.py
python src/train_yolo.py
python src/evaluate_yolo.py
streamlit run app/app.py
```

The committed dataset and binary model weights are not included in the repository; the data-preparation notebook is the starting point for regenerating them. Depending on the environment, run training/evaluation from `src/` or adjust imports/PYTHONPATH so the local `src` modules resolve correctly.

## Key learning outcomes

- Applied transfer learning and custom-head design across different CNN families.
- Built separate classification and detection data pipelines from COCO annotations.
- Used augmentation, freezing/unfreezing, schedulers, early stopping, and mixed precision.
- Implemented evaluation, model-history comparison, checkpoint loading, and interactive inference.
