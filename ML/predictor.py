# test predictor sample
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from ML.models.pytorch_mlp import PyTorchMLP

def predict_custom_images(model_path="saved_models/mlp-symbols.pt", test_dir="ML/data/custom_test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = PyTorchMLP(num_features=784, num_classes=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                   'add', 'dec', 'div', 'eq', 'mul', 'sub']
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Process all test images
    with torch.no_grad():
        for class_dir in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                
                # Load and transform image
                image = Image.open(img_path).convert("L")  # Convert to grayscale
                tensor_image = transform(image).unsqueeze(0).to(device)

                # Prediction
                output = model(tensor_image)
                _, pred = torch.max(output, 1)
                predicted_class = class_names[pred.item()]
                
                # Display results
                plt.imshow(image, cmap='gray')
                plt.title(f"True: {class_dir}\nPredicted: {predicted_class}")
                plt.axis('off')
                plt.show()

if __name__ == "__main__":
    predict_custom_images()