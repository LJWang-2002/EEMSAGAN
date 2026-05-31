import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import make_layer, default_init_weights


class CSAM(nn.Module):

    def __init__(self, gate_channels, reduction=4, no_spatial=False, spatial_kernel=7):
        super(CSAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels // reduction, gate_channels, 1, bias=False)
        )

        self.no_spatial = no_spatial
        if not no_spatial:
            padding = spatial_kernel // 2
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                          padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        if not self.no_spatial:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_in = torch.cat([avg_out, max_out], dim=1)
            spatial_att = self.sigmoid(self.spatial_conv(spatial_in))
            x = x * spatial_att

        return x


class MSFEB(nn.Module):

    def __init__(self, inplanes, planes):
        super(MSFEB, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=True),
            nn.Conv2d(planes, planes, 3, 1, 1)
        )
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)

        self.conv = nn.Conv2d(planes * 3, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)

        out = self.conv(torch.cat([out1, out2, out3], dim=1))
        out = self.lrelu(out)
        return out


class Hybrid_Attention_Fusion(nn.Module):

    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super(Hybrid_Attention_Fusion, self).__init__()
        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * channels // reduction, channels * 2, 1, bias=False)
        )

        self.weight_conv_h = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.weight_conv_l = nn.Conv2d(2 * channels, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, I_LR_hat, F_last):

        batch_size, channels, height, width = I_LR_hat.shape

        F_cat = torch.cat([I_LR_hat, F_last], dim=1)

        avg_out = self.channel_mlp(self.avg_pool(F_cat))
        max_out = self.channel_mlp(self.max_pool(F_cat))
        channel_feat = avg_out + max_out

        W_h_raw = self.weight_conv_h(channel_feat)
        W_l_raw = self.weight_conv_l(channel_feat)

        combined = torch.cat([W_h_raw, W_l_raw], dim=1)
        combined = combined.view(batch_size, channels, 2, 1, 1)
        weights = self.softmax(combined)

        W_h = weights[:, :, 0, :, :]
        W_l = weights[:, :, 1, :, :]

        F_fused = F_last * W_h + I_LR_hat * W_l

        avg_out = torch.mean(F_fused, dim=1, keepdim=True)
        max_out, _ = torch.max(F_fused, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)

        spatial_att = self.spatial_conv(spatial_in)
        spatial_att = self.sigmoid(spatial_att)

        I_SR = F_fused * spatial_att

        return I_SR


class PCAB(nn.Module):

    def __init__(self, in_channels, out_channels, pyconv_kernels=None, pyconv_groups=None,
                 stride=1, dilation=1, bias=False, attention=True,
                 reduction=4, spatial_kernel=7,
                 use_spatial=True):
        super(PCAB, self).__init__()

        if pyconv_groups is None:
            pyconv_groups = [1, 4, 8]
        if pyconv_kernels is None:
            pyconv_kernels = [3, 5, 7]

        self.attention = attention
        self.use_spatial = use_spatial

        split_channels = self._set_channels(out_channels, len(pyconv_kernels))

        self.pyconv_levels = nn.ModuleList([])
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels.append(nn.Conv2d(in_channels, split_channels[i],
                                                kernel_size=pyconv_kernels[i],
                                                stride=stride, padding=pyconv_kernels[i] // 2,
                                                groups=pyconv_groups[i],
                                                dilation=dilation, bias=bias))

        if use_spatial:
            self.attention_layers = nn.ModuleList([])
            for ch in split_channels:
                self.attention_layers.append(
                    CSAM(
                        gate_channels=ch,
                        reduction=reduction,
                        no_spatial=not use_spatial,
                        spatial_kernel=spatial_kernel
                    )
                )

    def forward(self, x):
        feas = [layer(x) for layer in self.pyconv_levels]

        if self.attention and hasattr(self, 'attention_layers'):
            out = []
            for i, feat in enumerate(feas):
                if i < len(self.attention_layers):
                    out.append(self.attention_layers[i](feat))
                else:
                    out.append(feat)
        else:
            out = feas

        out = torch.cat(out, dim=1)
        return out

    def _set_channels(self, out_channels, levels):
        if levels == 1:
            split_channels = [out_channels]
        elif levels == 2:
            split_channels = [out_channels // 2 for _ in range(2)]
        elif levels == 3:
            split_channels = [out_channels // 2, out_channels // 4, out_channels // 4]
        elif levels == 4:
            split_channels = [out_channels // 4 for _ in range(4)]
        else:
            raise NotImplementedError(f"no support: {levels}")
        return split_channels


class PCARDB(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32, reduction=4, use_spatial=True):
        super(PCARDB, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)

        pyramid_in_channels = num_feat + 4 * num_grow_ch

        self.pyramid_conv = PCAB(
            in_channels=pyramid_in_channels,
            out_channels=num_feat,
            pyconv_kernels=[3, 5, 7],
            pyconv_groups=[1, 4, 8],
            attention=True,
            reduction=reduction,
            use_spatial=use_spatial
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))

        pyramid_input = torch.cat((x, x1, x2, x3, x4), 1)
        x5 = self.pyramid_conv(pyramid_input)

        return x5 + x


class RAB(nn.Module):

    def __init__(self, num_feat, num_grow_ch=32, reduction=4, use_spatial=True):
        super(RAB, self).__init__()
        self.pcardb1 = PCARDB(num_feat, num_grow_ch, reduction, use_spatial)
        self.pcardb2 = PCARDB(num_feat, num_grow_ch, reduction, use_spatial)
        self.pcardb3 = PCARDB(num_feat, num_grow_ch, reduction, use_spatial)

        self.cs_attention = CSAM(
            gate_channels=num_feat,
            reduction=reduction,
            no_spatial=not use_spatial,
            spatial_kernel=7
        )

    def forward(self, x):
        out = self.pcardb1(x)
        out = self.pcardb2(out)
        out = self.pcardb3(out)

        out = self.cs_attention(out)

        return out + x


@ARCH_REGISTRY.register()
class EEMSANet(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=12,
                 num_grow_ch=32, reduction=4, use_spatial=True):
        super(EEMSANet, self).__init__()
        self.scale = 4
        self.reduction = reduction
        self.use_spatial = use_spatial

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.rab_body = make_layer(
            lambda: RAB(num_feat, num_grow_ch, reduction, use_spatial),
            num_block
        )

        self.rab_conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.global_cs_attention = CSAM(
            gate_channels=num_feat,
            reduction=reduction,
            no_spatial=not use_spatial,
            spatial_kernel=7
        )

        self.msfeb_path = MSFEB(num_in_ch, num_feat)

        self.hybrid_fusion = Hybrid_Attention_Fusion(
            channels=num_feat,
            reduction=4,
            spatial_kernel=7
        )

        self.upsample1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        self.upsample2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)


    def forward(self, x):
        rab_feat_init = self.lrelu(self.conv_first(x))
        rab_body_feat = self.rab_body(rab_feat_init)
        rab_body_feat = self.rab_conv_body(rab_body_feat)

        rab_body_feat = self.global_cs_attention(rab_body_feat)

        rab_feat = rab_feat_init + rab_body_feat

        msfeb_feat = self.msfeb_path(x)

        fused_feat = self.hybrid_fusion(msfeb_feat, rab_feat)

        up_feat1 = self.upsample1(fused_feat)

        up_feat2 = self.upsample2(up_feat1)

        out = self.conv_last(up_feat2)

        return out