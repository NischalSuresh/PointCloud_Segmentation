import torch
import torch.nn as nn
from tasks.semantic.model.vit import *

class trans_fuse(nn.Module):
    def __init__(self):
        super().__init__()
        image_size = [512,512]
        patch_size = 16
        self.patch_size = patch_size
        n_layers = 12
        d_model = 192
        self.d_model = d_model
        self.transformer_encoder_range = VisionTransformer(image_size,
            patch_size,
            n_layers,
            d_model = 192,
            d_ff = 4* d_model,
            n_heads = 6,
            n_cls = 20,
            dropout=0.1,
            drop_path_rate=0.0,
            distilled=False,
            channels=3,)
        self.transformer_encoder_zxy = VisionTransformer(image_size,
            patch_size,
            n_layers,
            d_model = 192,
            d_ff = 4* d_model,
            n_heads = 6,
            n_cls = 20,
            dropout=0.1,
            drop_path_rate=0.0,
            distilled=False,
            channels=3,)
        self.transformer_encoder_remission = VisionTransformer(image_size,
            patch_size,
            n_layers,
            d_model = 192,
            d_ff = 4* d_model,
            n_heads = 6,
            n_cls = 20,
            dropout=0.1,
            drop_path_rate=0.0,
            distilled=False,
            channels=3,)
        self.upsample = nn.UpsamplingBilinear2d(size= (64,2048))
        # self.transformer_encoder_zxy.load_state_dict(torch.load('/home/ubuntu/FPS-Net/train/tasks/semantic/trans_enc_state_dict.pt'))
        # self.transformer_encoder_range.load_state_dict(torch.load('/home/ubuntu/FPS-Net/train/tasks/semantic/trans_enc_state_dict.pt'))
        # self.transformer_encoder_remission.load_state_dict(torch.load('/home/ubuntu/FPS-Net/train/tasks/semantic/trans_enc_state_dict.pt'))
        # #-----#
        self.merge = nn.Sequential(nn.Conv2d(d_model * 3, 32, kernel_size=1, padding=0),
                            nn.BatchNorm2d(32),
                            nn.LeakyReLU())

    def forward(self, x):
        B, _, H, W = x.shape
        # pdb.set_trace()
        range = x[:,0,:,:].unsqueeze(1)
        zxy = x[:,1:4,:,:]
        remission = x[:,-1,:,:].unsqueeze(1)

        # edited to make the number of channels 3 across all modes of input
        range = range.repeat(1,3,1,1)
        remission = remission.repeat(1,3,1,1)

        range = self.transformer_encoder_range(range)[:,1:,:] #remove class token
        zxy = self.transformer_encoder_zxy(zxy)[:,1:,:] #remove class token
        remission = self.transformer_encoder_remission(remission)[:,1:,:] #remove class token
        # print("shape of range after transformer",range.shape)
        range = range.transpose(-1,-2).reshape(B,self.d_model,int(H/self.patch_size),int(W/self.patch_size))
        zxy = zxy.transpose(-1,-2).reshape(B,self.d_model,int(H/self.patch_size),int(W/self.patch_size))
        remission = remission.transpose(-1,-2).reshape(B,self.d_model,int(H/self.patch_size),int(W/self.patch_size))
        # print("shape of range after reshape",range.shape)
        range = self.upsample(range)
        zxy = self.upsample(zxy)
        remission = self.upsample(remission)
        # print("shape of range after upsampling=", range.shape)
        x = torch.cat((range, zxy, remission), dim=1)
        x = self.merge(x)
        # print("shape after merge=", x.shape)
        # pdb.set_trace()
        return x