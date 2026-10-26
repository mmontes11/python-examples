import torch
import torch.nn.functional as F    
from torch.autograd import grad

def tensors():
    tensor = torch.tensor([1 ,2, 3])
    print("tensor type: ", tensor.dtype)
    floattensor = tensor.to(torch.float32)
    print("tensor type: ", floattensor.dtype)

    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("2d tensor: ", tensor2d)
    print("2d tensor shape: ", tensor2d.shape)

    # .view() requires the original data to be contiguous 
    # and will fail if it isn’t
    print("2d tensor view: ", tensor2d.view(3, 2))

    # .reshape() will work regardless, copying the data 
    # if necessary to ensure the desired shape.)
    print("2d tensor reshape: ", tensor2d.reshape(3, 2))

    print("2d tensor transpose: ", tensor2d.T)
    print("2d tensor transpose at: ", tensor2d @ tensor2d.T)

    print("2d tensor multiplied by transpose: ", tensor2d.matmul(tensor2d.T)) 

def single_layer_neural_network():
    y = torch.tensor([1.0])         
    x1 = torch.tensor([1.1])   
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)         
    
    z = x1 * w1 + b                
    a = torch.sigmoid(z)              
    
    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)  
    grad_L_b = grad(loss, b, retain_graph=True)
    print(grad_L_w1)
    print(grad_L_b)

    # same as before, but higher level
    loss.backward()
    print(w1.grad)
    print(b.grad)

class MultiLayerNeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def multi_layer_neural_nework():
    model = MultiLayerNeuralNetwork(50, 3)
    print(model)

    num_params=0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print("Total number of trainable model parameters: ", num_params)
    # or shorter
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters (shorter): ", num_params)

    print("First layer: \n")
    print(model.layers[0].weight)
    print(model.layers[0].weight.shape)

    print("First layer bias: \n")
    print(model.layers[0].bias)
    print(model.layers[0].bias.shape)

    print("Init new neural network with seed")
    torch.manual_seed(123)
    model = MultiLayerNeuralNetwork(50, 3)
    print(model.layers[0].weight)

    print("Single forward pass")
    torch.manual_seed(123)
    X = torch.rand((1,50)) # training example
    out = model(X) # torch.nn.Module has a fixed __call__ that always calls the method named forward. 
    print(out)

    print("Signle forward pass without keeping track of gradients (cheaper):")
    with torch.no_grad():
        out = model(X)
    print(out)


def main():
    print("torch: ", torch.__version__)
    print("CUDA: ", torch.cuda.is_available())

    print("tensors:")
    tensors()

    print("single layer neural network:")
    single_layer_neural_network()

    print("multi layer neural network:")
    multi_layer_neural_nework()

if __name__ == "__main__":
    main()
