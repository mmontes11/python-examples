import torch

def main():
    print("torch: ", torch.__version__)
    print("CUDA: ", torch.cuda.is_available())

if __name__ == "__main__":
    main()
