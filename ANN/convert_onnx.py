import os
from mnist import  Net
import torch.onnx
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, TensorDataset

model_path = os.path.join(os.getcwd(), "models", "mnistpy")
model_path += ".pth"

test_data = datasets.MNIST(
    root='data',
    train = False,
    download=True,
    transform=transforms.ToTensor())

test_loader = DataLoader(test_data, batch_size=test_data.targets.size()[0])


mnist_model = Net()
mnist_model.load_state_dict(torch.load(model_path, weights_only=True))
mnist_model.eval()


images, labels = next(iter(test_loader))
with torch.no_grad():
    outputs = mnist_model(images)
    # _, predicted = torch.max(outputs, 1)
    
torch.onnx.export(
    mnist_model,
    images, 
    "mnist_model.onnx",
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=10,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput'], # the model's output names 
    dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                'modelOutput' : {0 : 'batch_size'}}
    )