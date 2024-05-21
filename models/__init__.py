from .mimounet import MIMOUNet, MIMOUNetPlus
from .dwt_unet import HMimoUnet

def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    elif model_name == "dwt":
        return HMimoUnet(in_channel=9)
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')