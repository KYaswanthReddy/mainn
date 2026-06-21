import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightImageEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(x + self.conv(x))

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=4, d_model=512):
        super().__init__()
        self.num_heads = num_heads
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.d_model = d_model
        
        self.q_proj = nn.Linear(img_dim, d_model)
        self.k_proj = nn.Linear(text_dim, d_model)
        self.v_proj = nn.Linear(text_dim, d_model)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, img_dim)
        self.norm = nn.LayerNorm(img_dim)
        
    def forward(self, img_feats, text_feats):
        """
        img_feats: (B, img_dim, H, W)
        text_feats: (B, text_dim)
        """
        B, C, H, W = img_feats.size()
        # Flatten image features to sequence: (B, H*W, C)
        img_flat = img_feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Add sequence dimension to text features: (B, 1, text_dim)
        text_flat = text_feats.unsqueeze(1)
        
        q = self.q_proj(img_flat)
        k = self.k_proj(text_flat)
        v = self.v_proj(text_flat)
        
        attn_out, _ = self.mha(q, k, v)
        attn_out = self.out_proj(attn_out)
        out = self.norm(img_flat + attn_out)
        
        # Reshape back to (B, C, H, W)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

class AdaLN3d(nn.Module):
    def __init__(self, n_channel, text_dim):
        super().__init__()
        self.n_channel = n_channel
        self.eps = 1e-5
        
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, n_channel * 2),
        )
        
        # Initialize to zero so it initially behaves like standard LayerNorm
        nn.init.zeros_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        
    def forward(self, x, text_features):
        """
        x: (B, C, L, H, W)
        text_features: (B, text_dim) or None
        """
        B, C, L, H, W = x.size()
        if text_features is None:
            text_features = torch.zeros(B, self.mlp[0].in_features, device=x.device)
            
        x_perm = x.permute(0, 2, 3, 4, 1)
        mean = x_perm.mean(dim=-1, keepdim=True)
        var = x_perm.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_perm - mean) / torch.sqrt(var + self.eps)
        
        scale_shift = self.mlp(text_features)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        
        # Bounding scale and shift to prevent gradient/value explosion
        scale = torch.clamp(scale, min=-0.9, max=2.0)
        shift = torch.clamp(shift, min=-3.0, max=3.0)
        
        # Reshape to (B, 1, 1, 1, C)
        scale = scale.view(B, 1, 1, 1, C)
        shift = shift.view(B, 1, 1, 1, C)
        
        out = (1.0 + scale) * x_norm + shift
        return out.permute(0, 4, 1, 2, 3)

def get_dataset_prompts(dataset_name, label_values, prompt_type):
    dataset_norm = dataset_name.lower()
    if 'pavia' in dataset_norm:
        pavia_ehsnet = {
            "tree": "A hyperspectral image patch containing tree vegetation with strong chlorophyll spectral response",
            "asphalt": "A hyperspectral image patch containing asphalt road surface with dark spectral characteristics",
            "brick": "A hyperspectral image patch containing brick building material with urban spectral properties",
            "bitumen": "A hyperspectral image patch containing bitumen surface with low reflectance characteristics",
            "shadow": "A hyperspectral image patch containing shadow regions with reduced spectral reflectance",
            "meadow": "A hyperspectral image patch containing meadow vegetation with dense grass spectral signatures",
            "bare soil": "A hyperspectral image patch containing exposed bare soil with earth surface spectral characteristics"
        }
        global_scene = "A hyperspectral image of Pavia University containing tree vegetation, asphalt roads, brick structures, bitumen surfaces, shadow regions, meadow vegetation, and bare soil."
        
        prompts = []
        for lbl in label_values:
            lbl_key = lbl.lower().strip()
            if lbl_key not in pavia_ehsnet:
                matched = False
                for k, v in pavia_ehsnet.items():
                    if k in lbl_key or lbl_key in k:
                        lbl_key = k
                        matched = True
                        break
                if not matched:
                    lbl_key = "tree" # fallback
            
            desc = pavia_ehsnet[lbl_key]
            if prompt_type == 'original':
                prompts.append(f"A hyperspectral remote sensing satellite image showing {lbl} land cover with distinct spectral signature")
            elif prompt_type == 'ehsnet':
                prompts.append(desc)
            elif prompt_type == 'rich':
                prompts.append(f"{global_scene} {desc}")
            else:
                prompts.append(desc)
        return prompts
    else:
        # Fallback for Houston or other datasets: generate prompts using templates
        prompts = []
        for lbl in label_values:
            if prompt_type == 'original':
                prompts.append(f"A hyperspectral remote sensing satellite image showing {lbl} land cover with distinct spectral signature")
            elif prompt_type == 'ehsnet':
                prompts.append(f"A hyperspectral image patch containing {lbl} with characteristic spectral response")
            elif prompt_type == 'rich':
                prompts.append(f"A hyperspectral image of the scene containing {lbl}. This patch shows {lbl} with distinct spectral signature.")
            else:
                prompts.append(f"A hyperspectral remote sensing satellite image showing {lbl} land cover with distinct spectral signature")
        return prompts

class TextFeatureCache:
    def __init__(self, clip_model, label_values, device, prompts=None, prompt_template=None):
        import clip as _clip
        if prompts is None:
            if prompt_template is None:
                prompt_template = (
                    "A hyperspectral remote sensing satellite image "
                    "showing {label} land cover with distinct spectral signature"
                )
            prompts = [prompt_template.format(label=lbl) for lbl in label_values]
            
        tokens = _clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens).float()
            text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features.detach()
        self.device = device

    def index_by_labels(self, labels):
        feats = self.text_features.to(labels.device)
        return feats[labels.long()]
