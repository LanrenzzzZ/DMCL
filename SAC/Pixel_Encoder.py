import torch
import numpy as np
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_channel=32, output_logits=False):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape           # [3, 84, 84]
        self.feature_dim = feature_dim      # 50
        self.num_layers = num_layers        # 4

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_channel, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_channel, num_channel, 3, stride=1))
        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers]
        self.FC = nn.Linear(num_channel * out_dim * out_dim, self.feature_dim)
        self.LN = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def forward_conv(self, obs):
        # obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)

        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)      # [B, 32, 35, 35]

        if detach:
            h = h.detach()
        h_fc = self.FC(h)               # [B, 50]
        self.outputs['fc'] = h_fc

        h_norm = self.LN(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, *args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}

def make_encoder(
        encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )




