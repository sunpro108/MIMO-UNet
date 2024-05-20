from models import build_net


model_name = "MIMO-UNet"
# model_name = "MIMO-UNetPlus"
model = build_net(model_name)
print(model.parameters())