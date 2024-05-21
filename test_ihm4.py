from data import train_dataloader
import torch
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

# if __name__ == "__main__":
#     loader_train = train_dataloader(subset='Hday2night')
#     data = next(iter(loader_train))
#     print(data.keys())

psnr = PSNR(data_range=(0.0, 1.0), reduction='none', dim=(1,2,3))
preds = torch.rand((16,3,256,256))
target = torch.rand((16,3,256,256))
res = psnr(preds, target)
print(res)