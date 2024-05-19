import random
from functools import partial

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)



class TripleCompose(transforms.Compose):
    def __call__(self, img_comp, img_real, img_mask):
        for t in self.transforms:
            img_comp, img_real, img_mask = t(img_comp, img_real, img_mask)
        return img_comp, img_real, img_mask


class TripleResize(transforms.Resize):
    def __call__(self, img_comp, img_real, img_mask):
        return super().__call__(img_comp), super().__call__(img_real), super().__call__(img_mask)


class TripleRandomCrop(transforms.RandomCrop):
    """
    Randomly crop the image.
    """
    def __init__(self, size:tuple):
        super().__init__()
        self.size = size

    def forward(self, img_comp, img_real, img_mask):
        assert img_comp.size == img_real.size == img_mask.size, 'RandomCrop: image size mismatch'
        step = 0
        # todo: make sure the sun mask is left after cropping
        # !while here is dangerous!
        while True:
            step+=1
            i, j, h, w = transforms.RandomCrop.get_params(img_mask, self.size)
            crop_func = partial(F.crop, top=i, left=j, height=h, width=w)
            croped_mask = crop_func(img_mask)
            if F.to_tensor(croped_mask).sum() > 1:
                return crop_func(img_comp), crop_func(img_real), crop_func(img_mask)
            if step > 10:
                print("warning: too much step to make sun mask left after cropping")


class TripleRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """
    Randomly horizontal Flip the image
    """
    def forward(self, img_comp, img_real, img_mask):
        assert img_comp.size == img_real.size == img_mask.size, 'RandomHorizontalFlip: image size mismatch'
        if random.random() > 0.5:
            return F.hflip(img_comp), F.hflip(img_real), F.hflip(img_mask)
        else:
            return img_comp, img_real, img_mask

class TripleToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, img_comp, img_real, img_mask):
        return F.to_tensor(img_comp), F.to_tensor(img_real), F.to_tensor(img_mask)


class TripleNormalize(transforms.Normalize):
    """
    Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    """
    def forward(self, img_comp, img_real, img_mask):
        return F.normalize(img_comp, self.mean, self.std), F.normalize(img_real, self.mean, self.std), img_mask

class TripleToPILImage(transforms.ToPILImage):
    """
    Convert a tensor or an ndarray to PIL Image.
    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.
    """
    def __call__(self, img_comp, img_real, img_mask):
        return F.to_pil_image(img_comp), F.to_pil_image(img_real), F.to_pil_image(img_mask, mode='L')