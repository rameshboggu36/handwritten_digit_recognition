import torch
from torchvision.transforms import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from myModel import MNISTmodel
from PIL import Image, ImageOps

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    model = torch.load(r"MNIST.pt")
    model.eval()
    target_layers = [model.conv1]
    cam = GradCAM(model=model, target_layers=target_layers)

    return model, cam

def main():
    # loading the cam model and MNIST_model 
    model, cam = load_model()
    
    st.header("Hand written Digit recognizer along with Heatmap by Ramesh")
    st.divider()
    uploaded_file = st.file_uploader("Upload the image to classify and also to visulaize it's heatmap")

    if uploaded_file is not None:
        # Reading the uploaded image into pil and then into torch.tensor format
        pil_img = ImageOps.grayscale(Image.open(uploaded_file).resize((28, 28)))
        torch_tensor = transforms.ToTensor()(pil_img).view(-1, 1, *pil_img.size)

        # generating the heat map using cam model
        gray_scale_cam = cam(input_tensor=torch_tensor)
        prediction = model(torch_tensor)
        
        in_img_vis = torch_tensor.permute(0, 2, 3, 1).numpy()  
        st.image(in_img_vis, width=480, caption='Uploaded Image')
        # this line combines the original image and the heat map generated using the cam model
        visualization = show_cam_on_image(in_img_vis, np.transpose(gray_scale_cam, (1, 2, 0)), image_weight=0.7)
        # st.write(visualization)
        conf = torch.max(prediction).item()
        pred = torch.argmax(prediction).item()
        st.write(f'The confidence score and the prediction is: {conf}, {pred}')
        if conf >= 0.5:
            st.image(Image.fromarray(visualization.squeeze()).resize((480, 480)),caption='Generated Image with Heatmap')
            st.success(f"The model has predicted this to be a {pred}", icon="✅")
        else:
            st.warning(f'Upload the correct image containing the number', icon="⚠️")

        
if __name__=="__main__":
    main()
