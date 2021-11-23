import torch
import torch.nn as nn
import torchvision.transforms as transform
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        channel_list = [3, 64, 128, 256, 512, 1024]
        self.encoding_blocks = nn.ModuleList([Block(channel_list[i], channel_list[i+1]) for i in range(len(channel_list)-1)])

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.encoding_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.channel_list = [1024, 512, 256, 128, 64]
        self.decoding_blocks = nn.ModuleList([nn.ConvTranspose2d(self.channel_list[i], self.channel_list[i+1], 2, 2) for i in range(len(self.channel_list)-1)])
        self.blocks = nn.ModuleList([Block(self.channel_list[i], self.channel_list[i+1]) for i in range(len(self.channel_list)-1)])

    def forward(self, x, encoder_features):
        for index in range(len(self.channel_list)-1):
            x = self.decoding_blocks[index](x)
            encoder_feature = self.crop(encoder_features[index], x)
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.blocks[index](x)
        return x

    def crop(self, encoding_feature, x):
        _, _, h, w = x.shape
        encoding_feature = transform.CenterCrop([h, w])(encoding_feature)
        return encoding_feature


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.end_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        encoding_features = self.encoder(x)
        out = self.decoder(encoding_features[-1], encoding_features[::-1][1:])
        out = self.end_conv(out)
        out = F.interpolate(out, (h, w))
        return out
