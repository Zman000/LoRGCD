from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except ImportError:  # Allows CPU smoke tests with torchvision-only environments.
    timm = None


class MLPHead(nn.Module):
    """DINO-style projection MLP used for representation contrastive losses."""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256, nlayers: int = 3):
        super().__init__()
        nlayers = max(1, int(nlayers))
        if nlayers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class CosineClassifier(nn.Module):
    """Cosine classifier equivalent in spirit to a normalized DINO last layer."""
    def __init__(self, dim: int, num_classes: int, scale: float = 1.0, learn_scale: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        if learn_scale:
            self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())
        else:
            self.register_buffer("log_scale", torch.tensor(float(scale)).log())

    def forward(self, z):
        z = F.normalize(z, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = self.log_scale.exp().clamp(0.1, 100.0)
        return scale * F.linear(z, w)


class GCDModel(nn.Module):
    """DINO backbone + projection MLP + parametric GCD classifier.

    `semantic` is the normalized backbone feature used by the category classifier and by
    LoRGCD geometry. `z` is the projection feature used for instance/supervised contrastive
    representation learning. This mirrors the separation in the official SimGCD design.
    """
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        proj_dim: int = 256,
        proj_hidden: int = 2048,
        proj_layers: int = 3,
        pretrained: bool = True,
        freeze_blocks: int = 11,
        classifier_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name.startswith("torchvision:"):
            self.backbone, feat_dim = self._build_torchvision_backbone(
                backbone_name.split(":", 1)[1], pretrained
            )
        else:
            if timm is None:
                raise ImportError(
                    "timm is required for DINO/timm backbones. Install requirements.txt, "
                    "or use torchvision:resnet18 for a dependency-light smoke test."
                )
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            feat_dim = getattr(self.backbone, "num_features", None)
            if feat_dim is None:
                raise ValueError(f"Could not infer feature dimension for backbone {backbone_name}")
        self.feat_dim = int(feat_dim)
        self.projector = MLPHead(self.feat_dim, proj_hidden, proj_dim, nlayers=proj_layers)
        self.classifier = CosineClassifier(self.feat_dim, num_classes, scale=classifier_scale)
        self._freeze_early_blocks(freeze_blocks)

    @staticmethod
    def _build_torchvision_backbone(name: str, pretrained: bool):
        from torchvision import models as tvm
        if not hasattr(tvm, name):
            raise ValueError(f"Unknown torchvision backbone: {name}")
        builder = getattr(tvm, name)
        weights = None
        if pretrained:
            try:
                weights_enum = tvm.get_model_weights(name)
                weights = weights_enum.DEFAULT
            except Exception:
                weights = "DEFAULT"
        model = builder(weights=weights)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            feat_dim = model.classifier.in_features
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"Torchvision backbone {name} needs an adapter for its classifier head")
        return model, int(feat_dim)

    def _freeze_early_blocks(self, freeze_blocks: int):
        if freeze_blocks <= 0:
            return
        if hasattr(self.backbone, "blocks"):
            if hasattr(self.backbone, "patch_embed"):
                for p in self.backbone.patch_embed.parameters():
                    p.requires_grad = False
            if hasattr(self.backbone, "pos_embed") and isinstance(self.backbone.pos_embed, nn.Parameter):
                self.backbone.pos_embed.requires_grad = False
            # Freeze everything first, then unfreeze requested late blocks and final norm.
            for p in self.backbone.parameters():
                p.requires_grad = False
            blocks = list(self.backbone.blocks)
            for block in blocks[min(freeze_blocks, len(blocks)):]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

    def encode(self, x):
        h = self.backbone(x)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.ndim > 2:
            h = h.mean(dim=1)
        semantic = F.normalize(h, dim=-1)
        z = F.normalize(self.projector(h), dim=-1)
        return h, semantic, z

    def forward(self, x):
        h, semantic, z = self.encode(x)
        logits = self.classifier(semantic)
        return {"features": h, "semantic": semantic, "z": z, "logits": logits}
