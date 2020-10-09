import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        point_feature = batch_dict['point_feature']
        coords = batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        lidar = batch_dict['points']
        # batch_dict: no voxels, only points with features
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            this_point_feature = point_feature[batch_idx]
            spatial_feature = torch.zeros(
                self.nz * self.num_bev_features,
                self.ny,
                self.nx, 
                dtype=this_point_feature.dtype,
                device=lidar.device)
            batch_mask = lidar[:, 0] == batch_idx
            this_lidar = lidar[batch_mask, :]  # [N, 5]: ixyzr
            x = this_lidar[:, 1]  # forward
            # x_np = x.detach().cpu().numpy()
            y = this_lidar[:, 2]  # left
            # y_np = y.detach().cpu().numpy()
            x_idx = (x-x.min()) // 0.16
            y_idx = (y-y.min()) // 0.16
            x_idx = torch.clamp(x_idx, 0, 431)
            y_idx = torch.clamp(y_idx, 0, 495)
            # maybe (ny, nx)
            x_idx = x_idx.type(torch.long)
            y_idx = y_idx.type(torch.long)
            spatial_feature[:, y_idx, x_idx] = this_point_feature
            batch_spatial_features.append(spatial_feature)
            plt.imshow(spatial_feature[:, :, :].detach().cpu().numpy().mean(axis=0), cmap=plt.cm.jet)
            plt.savefig('%s.jpg' % batch_idx)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_dict['spatial_features'] = batch_spatial_features
        
        return batch_dict
