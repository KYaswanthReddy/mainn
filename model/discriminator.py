import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size, dropout=0.0):
        super(discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.cls_head_src = nn.Linear(dim, num_classes)
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())
        # CLIP projection: branch directly from 512-dim feature (not 128-dim proj)
        self.clip_pro_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 512),
        )
        
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'): 

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        if self.dropout is not None:
            out3 = self.dropout(out3)
        out4 = self.relu4(self.fc2(out3))
        if self.dropout is not None:
            out4 = self.dropout(out4)
        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = self.pro_head(out4)
            proj_norm = F.normalize(proj)
            # CLIP projection from 512-dim feature directly (avoids 128-dim bottleneck)
            clip_proj = F.normalize(self.clip_pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj_norm, clip_proj

# model = discriminator(inchannel=176, outchannel=512, num_classes=12, patch_size=13)
# model = discriminator(inchannel=48, outchannel=512/, num_classes=7, patch_size=13)
# model = discriminator(inchannel=102 , outchannel=512, num_classes=7, patch_size=13)
# model.eval()
# input = torch.randn(1, 102, 13, 13)
# flops, params = profile(model, inputs=(input,))
# print(f"Parameters: {params / 1e6:.2f}M")
# print(f"FLOPs: {flops / 1e6:.2f}M")
