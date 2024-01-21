import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.models as models



class KeypointUpSample(nn.Module):
    def __init__(self, in_channels, num_keypoints):
        super().__init__()
        input_features = in_channels
        deconv_kernel = 4
        self.kps_score_lowres = nn.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        #nn.init.uniform_(self.kps_score_lowres.weight)
        #nn.init.uniform_(self.kps_score_lowres.bias)
        self.up_scale = 1
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        return torch.nn.functional.interpolate(
            x, scale_factor=float(self.up_scale), mode="bilinear", align_corners=False, recompute_scale_factor=False
        )




class SpatialSoftArgmax(nn.Module):
    """
    The spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.

    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, h, device=device),
                    torch.linspace(-1, 1, w, device=device),
                    indexing='ij',
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, h, device=device),
                torch.arange(0, w, device=device),
                indexing='ij',
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."

        # compute a spatial softmax over the input:
        # given an input of shape (B, C, H, W),
        # reshape it to (B*C, H*W) then apply
        # the softmax operator over the last dimension
        b, c, h, w = x.shape
        softmax = F.softmax(x.view(-1, h * w), dim=-1)

        # create a meshgrid of pixel coordinates
        # both in the x and y axes
        yc, xc = self._coord_grid(h, w, x.device)

        # element-wise multiply the x and y coordinates
        # with the softmax, then sum over the h*w dimension
        # this effectively computes the weighted mean of x
        # and y locations
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        # concatenate and reshape the result
        # to (B, C, 2) where for every feature
        # we have the expected x and y pixel
        # locations
        return torch.cat([x_mean, y_mean], dim=1).view(-1, c, 2)


class KeyPointSegNet(nn.Module):
    def __init__(self, args, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(KeyPointSegNet, self).__init__()

        self.args = args
        self.lim = lim

        k = args.n_kp

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"


        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()

        self.classifer = torch.nn.Sequential((list(deeplabv3_resnet50.children())[1]))



    def forward(self, img):
        input_shape = img.shape[-2:]

        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)

        # keypoint prediction branch
        heatmap = self.read_out(resnet_out) # (B, k, H//4, W//4)
        keypoints = self.spatialsoftargmax(heatmap)
        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.args.width // 2, self.args.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        # segmentation branch
        x = self.classifer(resnet_out)
        segout = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return keypoints, segout
    
    
class KeyPointSegNet_x(nn.Module):
    def __init__(self, args=None, lim=[-1., 1., -1., 1.], use_gpu=True):
        super(KeyPointSegNet_x, self).__init__()

        self.args = args
        self.lim = lim

        k = 7
        self.width = 640
        self.height = 480

        if use_gpu:
            self.device = "cuda"
        else:
            self.device = "cpu"


        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=True)
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes

        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])

        self.read_out = KeypointUpSample(2048, k)

        self.spatialsoftargmax = SpatialSoftArgmax()

        self.classifer = torch.nn.Sequential((list(deeplabv3_resnet50.children())[1]))



    def forward(self, img):
        input_shape = img.shape[-2:]

        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)

        # keypoint prediction branch
        heatmap = self.read_out(resnet_out) # (B, k, H//4, W//4)
        keypoints = self.spatialsoftargmax(heatmap)
        # mapping back to original resolution from [-1,1]
        offset = torch.tensor([self.lim[0], self.lim[2]], device = resnet_out.device)
        scale = torch.tensor([self.width // 2, self.height // 2], device = resnet_out.device)
        keypoints = keypoints - offset
        keypoints = keypoints * scale

        # segmentation branch
        x = self.classifer(resnet_out)
        segout = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return keypoints, segout
