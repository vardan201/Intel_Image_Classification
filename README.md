# Intel Image Classification

This project demonstrates end-to-end image classification on the Intel image dataset using a fine-tuned ResNet50 model trained in TensorFlow, with a Flask web app for deployment and real-time inference.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Training](#model-training)  
- [Data Augmentation](#data-augmentation)  
- [Hyperparameter Optimization](#hyperparameter-optimization)  
- [Model Saving](#model-saving)  
- [Flask Deployment](#flask-deployment)  
- [Usage](#usage)  
- [Requirements](#requirements)  
- [How to Run](#how-to-run)  
- [License](#license)

---

## Project Overview

- Fine-tune a ResNet50 model pretrained on ImageNet for 6-class Intel image classification:  
  `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.
- Extensive data augmentation techniques applied for improved generalization.
- Hyperparameter tuning using Optuna to find optimal model parameters.
- Model trained and saved in TensorFlow (`scenery_model.keras`).
- Flask app loads the saved model and provides a simple web interface for image upload and classification.

---

## Dataset

- Intel Image Classification Dataset (from Kaggle):  
  https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- Dataset is split into training and validation directories:
  - Training: `/content/seg_train/seg_train`
  - Validation: `/content/seg_test/seg_test`

---

## Model Training

- TensorFlow `tf.data` pipeline loads images resized to 224x224, batched (batch size 32).
- Uses `ImageDataGenerator`-like augmentation implemented with `tf.image` functions in a `map()` call.
- ResNet50 base model imported with pretrained ImageNet weights, top layer removed.
- Last few layers unfrozen for fine-tuning; rest frozen.
- Final layers include:
  - GlobalAveragePooling2D
  - Dense layer (units tuned by Optuna)
  - Batch Normalization
  - ReLU activation
  - Dropout (rate tuned by Optuna)
  - Output Dense layer with softmax activation for 6 classes
- Loss: sparse categorical crossentropy
- Optimizer: Adam (learning rate tuned)
- Early stopping used on validation accuracy.

---

## Data Augmentation

- Random horizontal flip
- Random brightness, contrast, saturation, hue changes
- Random zoom-in simulated with random crop + resize
- Cutout augmentation (random square patch masked to zero)
- Final preprocessing via `tf.keras.applications.resnet50.preprocess_input`

---

## Hyperparameter Optimization

- Using Optuna to tune:
  - Dense layer units (128, 256, 512)
  - Dropout rate (0.3 to 0.6)
  - Learning rate (1e-5 to 1e-3, log scale)
  - Number of unfrozen layers in ResNet50 (10 to 50)
  - L2 regularization weight (1e-5 to 1e-3, log scale)
- Early stopping after 2 epochs without improvement.
- Example best parameters found:
  - Dense units: 256
  - Dropout: ~0.34
  - Learning rate: 0.00025
  - Unfreeze layers: 10
  - L2 regularization: 0.00023

---

## Model Saving

- Model saved as `scenery_model.keras` after training.
- Compatible with TensorFlow's `load_model()` API for inference.

---

## Flask Deployment

- Flask app loads the saved model (`intel_model.keras` in deployment, adjust filename as needed).
- Defines classes: `['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']`.
- Web interface allows image upload via HTML form.
- Uploaded image is:
  - Opened with PIL
  - Resized to 224x224
  - Converted to numpy array and preprocessed with `preprocess_input`
  - Expanded dimensions to fit model input shape
- Model prediction run and highest probability class shown on page.
- Simple and clean UI with inline CSS styling.

---

## Usage

1. Start Flask app:

    ```bash
    python app.py
    ```

2. Open browser at:

    ```
    http://127.0.0.1:5000/
    ```

3. Upload an image to classify.

4. See prediction result instantly.

---

## Requirements

- Python 3.10  
- TensorFlow 2.19.0  
- Flask  
- NumPy  
- Pillow  

Install dependencies via:

```bash
pip install tensorflow flask numpy pillow optuna
