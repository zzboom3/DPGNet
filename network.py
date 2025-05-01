import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests
import math
from torch import nn
from torch.nn import init

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class GIE(nn.Module):

    def __init__(self, in_ch=64):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = self.fusion(torch.cat([b1, b2, b3], dim=1))
        return out + x  # 残差连接


class MAMRefinement(nn.Module):

    def __init__(self, dim=64):
        super().__init__()
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.ca(x) * x
        x = self.sa(x) * x
        dw = self.dw_conv(x)
        gate = self.gate(torch.cat([x, dw], dim=1))
        return gate * x + (1 - gate) * dw + residual


class ChannelAttention(nn.Module):

    def __init__(self, channel, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze())
        max_out = self.fc(self.max_pool(x).squeeze())
        out = avg_out + max_out
        return self.sigmoid(out.unsqueeze(-1).unsqueeze(-1))


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class MCG(nn.Module):

    def __init__(self, low_ch=64, skip_ch=64):
        super().__init__()
        self.gie = GIE(in_ch=low_ch)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(low_ch, skip_ch, 3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(skip_ch),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * skip_ch, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.GELU()
        )
        self.skip_conv = nn.Conv2d(skip_ch, skip_ch, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(skip_ch, skip_ch // 8, 1),
            nn.GELU(),
            nn.Conv2d(skip_ch // 8, skip_ch, 1),
            nn.Sigmoid()
        )
        self.refinement = MAMRefinement(dim=skip_ch)

    def forward(self, x_low, x_skip):
        x_enhanced = self.gie(x_low)
        x_up = self.upsample(x_enhanced)

        fused = self.fusion(torch.cat([x_up, x_skip], dim=1))
        fused = fused + self.skip_conv(x_skip)

        return self.refinement(fused)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    

INNER_DIM = 512


class MultiScaleConvModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depth_conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                       padding=3 // 2, groups=in_channels)
        self.depth_conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5,
                                       padding=5 // 2, groups=in_channels)
        self.depth_conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7,
                                       padding=7 // 2, groups=in_channels)

        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.fusion_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.GELU(), 
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        identity = x
        x = (self.depth_conv3x3(x) + self.depth_conv5x5(x) + self.depth_conv7x7(x)) / 3.0
        x = self.pointwise_conv(x) + identity 
        return self.fusion_proj(x) + identity


class Mamba_Enhanced_Adapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, INNER_DIM),
            nn.GELU(),
            nn.Linear(INNER_DIM, INNER_DIM)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(INNER_DIM, INNER_DIM),
            nn.GELU(),
            nn.Linear(INNER_DIM, input_dim)
        )

        self.multi_scale_adapter = MultiScaleConvModule(INNER_DIM)

        self.pre_norm = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(input_dim)
        self.feature_dropout = nn.Dropout(0.1)

        self.norm_scale = nn.Parameter(torch.ones(input_dim) * 1e-6)
        self.residual_scale = nn.Parameter(torch.ones(input_dim))
        self.channel_scale = nn.Parameter(torch.ones(1, 1, input_dim))  # 通道级缩放

    def forward(self, input_tensor):
        original_shape = input_tensor.shape
        batch_size, channels, height, width = original_shape

        seq_input = input_tensor.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        residual = seq_input

        normalized = self.pre_norm(seq_input)
        scaled_features = normalized * self.norm_scale * self.channel_scale + seq_input * self.residual_scale

        hidden_states = self.input_proj(scaled_features)  # [B, H*W, INNER_DIM]

       
        spatial_features = hidden_states.reshape(batch_size, height, width, INNER_DIM).permute(0, 3, 1, 2)
        processed_features = self.multi_scale_adapter(spatial_features)

        seq_features = processed_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, INNER_DIM)

        output = self.output_proj(seq_features)
        output = self.post_norm(output)

        final_output = (residual + output).reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return self.feature_dropout(final_output)


class DPGNet(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(DPGNet, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        self.mamba_model = AutoModel.from_pretrained("nvidia/MambaVision-L3-512-21K", trust_remote_code=True) 
        for param in self.mamba_model.parameters():
            param.requires_grad = False

        self.ma1 = Mamba_Enhanced_Adapter(256)
        self.ma2 = Mamba_Enhanced_Adapter(512)
        self.ma3 = Mamba_Enhanced_Adapter(1024)
        self.ma4 = Mamba_Enhanced_Adapter(2048)
            
        self.rfb1 = RFB_modified(400, 64)
        self.rfb2 = RFB_modified(800, 64)
        self.rfb3 = RFB_modified(1600, 64)
        self.rfb4 = RFB_modified(3200, 64)
    
        self.md1 = MCG(64, 64)
        self.md2 = MCG(64, 64)
        self.md3 = MCG(64, 64)
        self.md4 = MCG(64, 64)
        
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1_enc, x2_enc, x3_enc, x4_enc = self.encoder(x)
        _, featrue = self.mamba_model(x) 
        
        x1_mam = self.ma1(featrue[0])
        x2_mam = self.ma2(featrue[1])
        x3_mam = self.ma3(featrue[2])
        x4_mam = self.ma4(featrue[3])
        
        x1 = torch.cat([x1_enc, x1_mam], dim=1)    
        x2 = torch.cat([x2_enc, x2_mam], dim=1)   
        x3 = torch.cat([x3_enc, x3_mam], dim=1)       
        x4 = torch.cat([x4_enc, x4_mam], dim=1)    
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.md1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.md2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.md3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out, out1, out2


if __name__ == "__main__":
    with torch.no_grad():
        model = DPGNet().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)