import torch
import torch.nn as nn
from pcdet.utils import point_transform


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

idx = '000000'
ri, theta_idx, phi_idx = point_transform.get_range_image('000000')  # theta_idx, phi_idx: per point coordinate
lidar = point_transform.get_lidar(idx)
ri = torch.from_numpy(ri)
ri = ri.unsqueeze(0)
model = RangeImage2DBackbone()
o = model(ri)  # 64, 64, 512  RangeRCNN: 48 × 512
# RV-PV
feature_pv = o[:, :, theta_idx, phi_idx]
a = 10
# PV-BEV
# feature_pv (64*N)
N = feature_pv.shape[1]


def forward(self, batch_dict, **kwargs):
    # voxels: [M, max_points, ndim] float tensor. only contain points.
    # coordinates: [M, 3] int32 tensor. zyx format.
    # use voxelize then pfe then scatter

    # after pfe: torch.Size([13110, 64]), [M, num_features]
    # use mean VFE(avg of 32 points in voxels)
    voxel_features, coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
    batch_spatial_features = []
    batch_size = coords[:, 0].max().int().item() + 1
    num_bev_features = 64

    for batch_idx in range(batch_size):
        spatial_feature = torch.zeros(
            num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        batch_mask = coords[:, 0] == batch_idx
        this_coords = coords[batch_mask, :]
        # 
        indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # zyx
        indices = indices.type(torch.long)
        voxels = voxel_features[batch_mask, :]
        voxels = voxels.t()
        spatial_feature[:, indices] = voxels  # put feature of each voxel to a line
        batch_spatial_features.append(spatial_feature)

    batch_spatial_features = torch.stack(batch_spatial_features, 0)
    # to bev
    batch_spatial_features = batch_spatial_features.view(batch_size, num_bev_features * self.nz, self.ny, self.nx)
    batch_dict['spatial_features'] = batch_spatial_features
    # bev_backbone
    # feature_pv