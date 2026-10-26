import torch
import torch.nn.functional as F    
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

class ToyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.features= X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0];

def create_train_loader() -> DataLoader:
    torch.manual_seed(123)
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    train_ds = ToyDataset(X_train, y_train)

    return DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

def data_loader(train_loader: DataLoader):
    train_loader = create_train_loader()

    print("Iterating over train loader:")
    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx+1}:", x, y)


def training_loop(train_loader: DataLoader):
    torch.manual_seed(123)
    model = MultiLayerNeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(
      model.parameters(),
      lr=0.5
    )

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}"
            )
        
        model.eval()

def main():
    print("torch: ", torch.__version__)
    print("CUDA: ", torch.cuda.is_available())

    print("tensors:")
    tensors()

    print("single layer neural network:")
    single_layer_neural_network()

    print("multi layer neural network:")
    multi_layer_neural_nework()

    print("data loader:")
    train_loader = create_train_loader()
    data_loader(train_loader)

    print("training loop:")
    training_loop(train_loader)

if __name__ == "__main__":
    main()
