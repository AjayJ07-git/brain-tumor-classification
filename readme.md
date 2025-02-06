
# Brain Tumor Detection using ResNet-50

This repository contains a deep learning-based web application for **brain tumor classification** using a **ResNet-50** model. The model classifies MRI images into four categories: **Glioma, Meningioma, No Tumor, and Pituitary**.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Upload an MRI image (**JPG, JPEG, PNG**) and get a classification result.
- Uses **ResNet-50 with transfer learning** for accurate predictions.
- Web-based interface built with **Streamlit**.
- Supports **CPU and GPU (CUDA)** for inference.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AjayJ07-git/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Ensure you have the trained model file (`brain_tumor_resnet50_transfer_learning.pth`) in the project directory.

## Usage

To run the Streamlit web application:
```sh
streamlit run app.py
```
This will launch a web interface where you can upload an MRI scan image and receive a classification result.

## Model Details
- The model is based on **ResNet-50** with a modified fully connected layer.
- It has been trained on a brain tumor dataset with **four classes**:
  - **Glioma**
  - **Meningioma**
  - **No Tumor**
  - **Pituitary**
- The model uses **PyTorch** for deep learning inference.

## Dataset
The model is trained using an MRI-based brain tumor dataset. If you are interested in the dataset, you can check publicly available datasets like:
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Technologies Used
- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **Streamlit**
- **PIL (Python Imaging Library)**

## Project Structure
```
Brain-Tumor-Detection/
│── app.py                        # Streamlit web application
│── Brain_Tumor_Detection.ipynb    # Jupyter Notebook for model training
│── brain_tumor_resnet50_transfer_learning.pth  # Trained model file
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation
```

## Contributing
Contributions are welcome! If you find any issues or want to add improvements, feel free to open an **issue** or **pull request**.

