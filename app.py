import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Parameters
IMG_SIZE = 224  # The size to which the input images will be resized.
NUM_CLASSES = 4  # Number of output classes for the model (glioma, meningioma, no tumor, pituitary).
MODEL_PATH = "brain_tumor_resnet50_transfer_learning.pth"  # Path to the pre-trained model.

# Define transformations
# Preprocessing applied to each input image before passing to the model
input_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),  # Resize the image to the target size
    transforms.CenterCrop(IMG_SIZE),  # Crop the center of the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet's statistics
])

# Load model function
def load_model(model_path, device):
    """
    Load a pre-trained ResNet-50 model, modify the fully connected layer for the 4-class output,
    and load the model weights from the specified path.

    Args:
        model_path (str): Path to the trained model.
        device (torch.device): The device to load the model on ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The modified model ready for inference.
    """
    model = models.resnet50(pretrained=False)  # Load ResNet50 without pre-trained weights
    num_features = model.fc.in_features  # Get the number of input features for the final layer
    model.fc = torch.nn.Sequential(  # Modify the fully connected (fc) layer
        torch.nn.Linear(num_features, 512),  # First FC layer with 512 units
        torch.nn.ReLU(),  # ReLU activation
        torch.nn.Dropout(0.4),  # Dropout for regularization
        torch.nn.Linear(512, NUM_CLASSES)  # Final layer to output NUM_CLASSES
    )
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model.to(device).eval()  # Move model to device (CPU/GPU) and set it to evaluation mode
    return model

# Prediction function
def predict_image(image, model, device, class_names):
    """
    Predict the class of the given image using the loaded model.

    Args:
        image (PIL.Image): The input image to predict.
        model (torch.nn.Module): The trained model for prediction.
        device (torch.device): The device to perform the computation on.
        class_names (list): List of class names corresponding to the output labels.

    Returns:
        str: The predicted class label.
    """
    transformed_image = input_transforms(image).unsqueeze(0).to(device)  # Apply transformations and move image to device
    with torch.no_grad():  # Disable gradient calculations during inference
        outputs = model(transformed_image)  # Pass image through the model
    return class_names[torch.argmax(outputs).item()]  # Return the class with the highest probability

# Main application
def main():
    """
    The main function to launch the Streamlit app for brain tumor classification. It allows users
    to upload MRI images and displays the prediction result for the class of the tumor.
    """
    st.title("Brain Tumor Classification App")  # Title for the app
    st.markdown("""
        **Class Categories:**
        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary
    """)  # Description of the classes

    # Initialize session state for upload counter
    # This ensures the user can upload multiple files by resetting the uploader each time
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0  # Initialize upload counter to track uploads

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    model = load_model(MODEL_PATH, device)  # Load the trained model
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Class names for prediction

    # Dynamic file uploader key
    # This ensures that the upload widget is reset every time a new file is uploaded
    upload_key = f"upload_{st.session_state.upload_counter}"
    uploaded_file = st.file_uploader(
        "Upload MRI Image (JPG/JPEG/PNG)",  # Instruction for file upload
        type=["jpg", "jpeg", "png"],  # Accepted file formats
        key=upload_key  # Unique key for each upload
    )

    if uploaded_file is not None:
        # Display and process image
        image = Image.open(uploaded_file).convert("RGB")  # Open the uploaded image
        st.image(image, width=150, caption="Uploaded Image")  # Display the image with a caption
        prediction = predict_image(image, model, device, class_names)  # Get prediction for the image
        st.success(f"**Prediction:** {prediction.capitalize()}")  # Display the predicted class

        # Next prediction button
        if st.button("Next Prediction"):
            # Increment counter to reset uploader for the next prediction
            st.session_state.upload_counter += 1
            st.rerun()  # Rerun the app to reset the file uploader

if __name__ == "__main__":
    main()  # Run the app when the script is executed
