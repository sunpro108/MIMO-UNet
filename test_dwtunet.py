import torch 

from models import HMimoUnet

if __name__ == "__main__":
    x = torch.rand((8,3,256,256),dtype=torch.float32)
    model = HMimoUnet(in_channel=9)
    y = model(x)
    print(len(y))
    print(y[0].shape)