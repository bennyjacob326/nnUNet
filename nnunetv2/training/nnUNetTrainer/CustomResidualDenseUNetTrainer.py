#In this implementation, each residual block is followed by a dense block, 
#and the output of the dense block is concatenated with the 
#input of the subsequent block
import torch
from torch import nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer(torch.cat(outputs, 1)))
        return torch.cat(outputs, 1)


class CustomResidualDenseUNetTrainer(nn.Module):
    @staticmethod
    def build_network_architecture(plans_manager, dataset_json, configuration_manager, num_input_channels,
                                   enable_deep_supervision=True):
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = CustomResidualDenseUNetTrainer.convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        # Customize your network architecture here
        model = CustomResidualDenseUNet(input_channels=num_input_channels,
                                         n_stages=num_stages,
                                         features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                                                 configuration_manager.unet_max_num_features)
                                                             for i in range(num_stages)],
                                         conv_op=conv_op,
                                         kernel_sizes=configuration_manager.conv_kernel_sizes,
                                         strides=configuration_manager.pool_op_kernel_sizes,
                                         num_classes=label_manager.num_segmentation_heads,
                                         deep_supervision=enable_deep_supervision,
                                         **configuration_manager.__dict__)

        return model


class CustomResidualDenseUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage,
                 conv_op,
                 kernel_sizes,
                 strides,
                 num_classes: int,
                 deep_supervision: bool = False,
                 block: nn.Module = BasicBlockD,  # Change this to BasicBlockD or BottleneckD
                 dense_blocks_per_stage: int = 2,  # Number of dense blocks per stage
                 **kwargs):
        super().__init__()

        self.encoder = ResidualEncoder(input_channels=input_channels,
                                       n_stages=n_stages,
                                       features_per_stage=features_per_stage,
                                       conv_op=conv_op,
                                       kernel_sizes=kernel_sizes,
                                       strides=strides,
                                       block=block,
                                       **kwargs)

        # Add dense blocks
        self.dense_blocks = nn.ModuleList([
            DenseBlock(num_layers=3,  # Number of layers in each dense block
                       in_channels=features_per_stage[i],
                       growth_rate=32)
            for i in range(len(features_per_stage))
        ])

    def forward(self, x):
        skips = self.encoder(x)
        
        # Pass through dense blocks
        for i, skip in enumerate(skips):
            dense_out = self.dense_blocks[i](skip)
            if i < len(skips) - 1:
                skips[i + 1] = torch.cat([skips[i + 1], dense_out], dim=1)

        return skips

    @staticmethod
    def convert_dim_to_conv_op(dim):
        if dim == 1:
            return nn.Conv1d
        elif dim == 2:
            return nn.Conv2d
        elif dim == 3:
            return nn.Conv3d
        else:
            raise ValueError("Unsupported dimensionality. Only 1D, 2D, and 3D convolutions are supported.")
