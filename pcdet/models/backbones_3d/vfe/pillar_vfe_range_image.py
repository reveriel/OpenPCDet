import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate
from pcdet.utils import point_transform 
import matplotlib.pyplot as plt

class RangeNetEncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        mid_channels = out_channels // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, 3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels*3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        # self.dropout = nn.Dropout2d()

        if downsample:
            self.pooling = nn.MaxPool2d(2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.in_conv(x)

        x1 = self.atrous1(x)
        x2 = self.atrous2(x1)
        x3 = self.atrous3(x2)
        fused = self.fuse(torch.cat([x1, x2, x3], dim=1))

        # out = self.dropout(fused)
        out = fused
        if self.downsample:
            out = self.pooling(out)

        return out


class RangeNetDecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, skip_channels=None):
        super().__init__()
        self.upsample = upsample
        self.fuse_skip = bool(skip_channels)
        mid_channels = out_channels
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 2, 2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 3, 3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels*3,
                      skip_channels if self.fuse_skip else out_channels, 1),
            nn.BatchNorm2d(skip_channels if self.fuse_skip else out_channels),
            nn.ReLU()
        )
        # self.dropout = nn.Dropout2d()

        if upsample:
            self.interpolation = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)

        if skip_channels is not None:
            self.out_conv = nn.Conv2d(skip_channels * 2, out_channels, 1)

    def forward(self, x, skipped=None):
        x = self.in_conv(x)
        x1 = self.atrous1(x)
        x2 = self.atrous2(x1)
        x3 = self.atrous3(x2)
        fused = self.fuse(torch.cat([x1, x2, x3], dim=1))

        # out = self.dropout(fused)
        out = fused
        if self.upsample:
            out = self.interpolation(out)
        if self.fuse_skip:
            out = self.out_conv(torch.cat([out, skipped], dim=1))

        return out


class RangeImage2DBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 5
        encoder_channels = [32, 64, 128, 256, 256, 256]
        encoder_downsample = [False] * 2 + [True] * 4
        decoder_channels = [128, 128, 64, 64]
        decoder_upsample = [True] * 4
        self.encoders = nn.ModuleList([])
        self.encoders.append(RangeNetEncodingBlock(
            in_channels, encoder_channels[0], downsample=False))
        for i in range(len(encoder_channels)-1):
            self.encoders.append(
                RangeNetEncodingBlock(
                    encoder_channels[i], encoder_channels[i+1], encoder_downsample[i+1])
            )
        self.decoders = nn.ModuleList([])
        decoder_channels.insert(0, encoder_channels[-1])
        for i in range(len(decoder_channels)-1):
            self.decoders.append(
                RangeNetDecodingBlock(
                    decoder_channels[i], decoder_channels[i+1], decoder_upsample[i], encoder_channels[-(i+2)])
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoded_features = []
        for m in self.encoders:
            x = m(x)
            encoded_features.append(x)

        for m, f in zip(self.decoders, encoded_features[-2:0:-1]):
            x = m(x, f)  # x and f are concated

        return x

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)
        self.range_image_backbone = RangeImage2DBackbone()
        self.use_norm = self.model_cfg.USE_NORM

    def get_output_feature_dim(self):
        return 4

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        range_image = batch_dict['range_image']  # [B, C, H, W]
        plt.imshow(range_image[0, 3, :, :].cpu().numpy(), cmap=plt.cm.jet)
        plt.savefig('range_image.jpg')
        theta_idx = batch_dict['theta_idx']
        phi_idx = batch_dict['phi_idx']
        batch_size = batch_dict['batch_size']
        range_image_feature = self.range_image_backbone(range_image)
        batch_feature = []
        for batch_idx in range(batch_size):
            this_range_image_feature = range_image_feature[batch_idx]
            this_theta_idx = theta_idx[batch_idx]
            this_phi_idx = phi_idx[batch_idx]
            # RV-PV
            feature_pv = this_range_image_feature[:, this_theta_idx, this_phi_idx]
            batch_feature.append(feature_pv)

        batch_dict['point_feature'] = batch_feature
        # return [M, num_features]
        return batch_dict
