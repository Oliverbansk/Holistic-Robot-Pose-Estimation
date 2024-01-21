import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from .backbones.HRnet import get_hrnet
from .backbones.Resnet import get_resnet
from torch.nn import functional as F


class RootNet(nn.Module):

    def __init__(self, backbone, pred_xy=False, use_offset=False, add_fc=False, input_shape=(256,256),  **kwargs):
        
        super(RootNet, self).__init__()
        self.backbone_name = backbone
        if backbone in ["resnet34", "resnet50", "resnet"]:
            self.backbone = get_resnet(backbone)
            self.inplanes = self.backbone.block.expansion * 512
        elif backbone in ["hrnet", "hrnet32"]:
            self.backbone = get_hrnet(type_name=32, num_joints=7, depth_dim=1,
                                      pretrain=True, generate_feat=True, generate_hm=False)
            self.inplanes = 2048
        else:
            raise NotImplementedError
            
        self.pred_xy = pred_xy
        self.add_fc = add_fc
        self.use_offset = use_offset
        self.input_shape = input_shape
        self.output_shape = (input_shape[0]//4, input_shape[1]//4)
        self.outplanes = 256
        
        if self.pred_xy:
            self.deconv_layers = self._make_deconv_layer(3)
            self.xy_layer = nn.Conv2d(
                in_channels=self.outplanes,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0
            )
            
        if self.add_fc:
            self.depth_relu = nn.ReLU()
            self.depth_fc1 = nn.Linear(self.inplanes, self.inplanes//2)
            self.depth_bn1 = nn.BatchNorm1d(self.inplanes//2)
            self.depth_fc2 = nn.Linear(self.inplanes//2, self.inplanes//4)
            self.depth_bn2 = nn.BatchNorm1d(self.inplanes//4)
            self.depth_fc3 = nn.Linear(self.inplanes//4, self.inplanes//4)
            self.depth_bn3 = nn.BatchNorm1d(self.inplanes//4)
            self.depth_fc4 = nn.Linear(self.inplanes//4, self.inplanes//2)
            self.depth_bn4 = nn.BatchNorm1d(self.inplanes//2)
            self.depth_fc5 = nn.Linear(self.inplanes//2, self.inplanes)
            
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0
        )
        if self.use_offset:
            self.offset_layer = nn.Conv2d(
                in_channels=self.inplanes,
                out_channels=1, 
                kernel_size=1,
                stride=1,
                padding=0
            )

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        if self.backbone_name in ["resnet34", "resnet50", "resnet"]:
            fm = self.backbone(x)
            img_feat = torch.mean(fm.view(fm.size(0), fm.size(1), fm.size(2)*fm.size(3)), dim=2) # global average pooling
        elif self.backbone_name in ["hrnet", "hrnet32"]:
            img_feat = self.backbone(x)

        # x,y
        if self.pred_xy:
            xy = self.deconv_layers(fm)
            xy = self.xy_layer(xy)
            xy = xy.view(-1,1,self.output_shape[0]*self.output_shape[1])
            xy = F.softmax(xy,2)
            xy = xy.view(-1,1,self.output_shape[0],self.output_shape[1])
            hm_x = xy.sum(dim=(2))
            hm_y = xy.sum(dim=(3))
            coord_x = hm_x * torch.arange(self.output_shape[1]).float().cuda()
            coord_y = hm_y * torch.arange(self.output_shape[0]).float().cuda()
            coord_x = coord_x.sum(dim=2)
            coord_y = coord_y.sum(dim=2)

        # z
        if self.add_fc:
            img_feat1 = self.depth_relu(self.depth_bn1(self.depth_fc1(img_feat)))
            img_feat2 = self.depth_relu(self.depth_bn2(self.depth_fc2(img_feat1)))
            img_feat3 = self.depth_relu(self.depth_bn3(self.depth_fc3(img_feat2)))
            img_feat4 = self.depth_relu(self.depth_bn4(self.depth_fc4(img_feat3)))
            img_feat5 = self.depth_fc5(img_feat4)
            img_feat = img_feat + img_feat5 
        img_feat = torch.unsqueeze(img_feat,2)
        img_feat = torch.unsqueeze(img_feat,3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        depth = gamma * k_value.view(-1,1)
        
        if self.use_offset:
            offset = self.offset_layer(img_feat)
            offset = offset.view(-1,1) # unit: m
            offset *= 1000.0
            depth += offset

        if self.pred_xy:
            coord = torch.cat((coord_x, coord_y, depth), dim=1)
            return coord
        else:
            return depth

    def init_weights(self):
        if self.pred_xy:
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.xy_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            print("Initialized deconv and xy layer of RootNet.")
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        print("Initialized depth layer of RootNet.")
        if self.use_offset:
            for m in self.offset_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            print("Initialized offset layer of RootNet.")


def get_rootnet(backbone, pred_xy=False, use_offset=False, add_fc=False, input_shape=(256,256), **kwargs):
    model = RootNet(backbone, pred_xy, use_offset, add_fc, input_shape=(256,256), **kwargs)
    model.init_weights()
    return model


