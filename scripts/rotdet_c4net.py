import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utility (C4 + quadrati)
# =========================
def rot90_k(w, k: int):
    """Ruota kernel 2D di k*90° (dims ultimi due)."""
    k %= 4
    return w if k == 0 else torch.rot90(w, k, dims=(-2, -1))

def split_orient(x):
    """(B, C*4, H, W) -> (B, C, 4, H, W)"""
    B, C4, H, W = x.shape
    assert C4 % 4 == 0, "Il numero canali deve essere multiplo di 4."
    C = C4 // 4
    return x.view(B, C, 4, H, W)

def merge_orient(x):
    """(B, C, 4, H, W) -> (B, C*4, H, W)"""
    B, C, G, H, W = x.shape
    assert G == 4, "Mi aspetto 4 orientazioni (C4)."
    return x.view(B, C * 4, H, W)

def group_pool_c4(x, reduce="mean"):
    """Pooling sulle orientazioni: (B, C*4, H, W) -> (B, C, H, W)."""
    x = split_orient(x)  # (B, C, 4, H, W)
    if reduce == "mean":
        x = x.mean(dim=2)
    elif reduce == "max":
        x, _ = x.max(dim=2)
    else:
        raise ValueError("reduce deve essere 'mean' o 'max'")
    return x

# ============================================
# 1) STEM: Lifting Z2 -> C4 (assume H=W)
# ============================================
class LiftingConvC4(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=2, bias=True):
        super().__init__()
        pad = k // 2  # 'same' su quadrati
        self.k = k
        self.stride = stride
        self.pad = pad
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, k, k) * (2.0 / (in_ch * k * k)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W, "Per semplicità, uso input quadrati (H=W)."
        ys = []
        for r in range(4):
            w_r = rot90_k(self.weight, r)
            y_r = F.conv2d(x, w_r, bias=self.bias, stride=self.stride, padding=self.pad)
            ys.append(y_r)
        return torch.cat(ys, dim=1)  # (B, out_ch*4, H', W')

# =========================================================
# 2) GROUP CONV C4: C4 -> C4 (pesi vincolati, quadrati)
# =========================================================
class GroupConvC4(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, bias=True):
        super().__init__()
        pad = k // 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.stride = stride
        self.pad = pad

        # peso base indicizzato dall’orientazione relativa t ∈ {0..3}
        self.weight_base = nn.Parameter(
            torch.randn(out_ch, in_ch, 4, k, k) * (2.0 / (in_ch * k * k)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def build_full_weight(self, device=None):
        Cout, Cin, k = self.out_ch, self.in_ch, self.k
        blocks = []
        for r in range(4):                # orientazione di uscita
            row = []
            for s in range(4):            # orientazione di ingresso
                t = (s - r) % 4           # orientazione relativa
                W_t = self.weight_base[:, :, t, :, :]     # (Cout, Cin, k, k)
                W_rt = rot90_k(W_t, r)                     # ruota di r*90°
                row.append(W_rt)                           # (Cout, Cin, k, k)
            row = torch.cat(row, dim=1)                   # (Cout, Cin*4, k, k)
            blocks.append(row)
        W_full = torch.cat(blocks, dim=0)                  # (Cout*4, Cin*4, k, k)
        return W_full.to(device) if device is not None else W_full

    def forward(self, x):
        # x: (B, in_ch*4, H, W) con H=W
        B, C4, H, W = x.shape
        assert H == W, "Per semplicità, uso feature map quadrate (H=W)."
        assert C4 == self.in_ch * 4
        W = self.build_full_weight(device=x.device)
        b = None if self.bias is None else self.bias.repeat(4)
        return F.conv2d(x, W, bias=b, stride=self.stride, padding=self.pad)

# =======================================
# 3) Blocchi semplici: ConvC4 + BN + ReLU
# =======================================
class C4Block(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.conv = GroupConvC4(in_ch, out_ch, k=k, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

# =======================================
# 4) Head: invariance (classif) o equivariance
# =======================================
class InvariantHeadC4(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        # x: (B, in_ch*4, H, W)
        x = group_pool_c4(x, reduce="mean")  # (B, in_ch, H, W)
        x = self.gap(x).flatten(1)           # (B, in_ch)
        return self.fc(x)                    # (B, num_classes)

class EquivariantHeadC4(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv1x1 = GroupConvC4(in_ch, out_ch, k=1, stride=1)

    def forward(self, x):
        return self.conv1x1(x)               # (B, out_ch*4, H, W)

# =======================================
# 5) Modello: STEM + BACKBONE + HEAD
# =======================================
class C4Net(nn.Module):
    """
    Assunzioni per semplicità/leggibilità:
      - input quadrati (H=W), idealmente multipli di 4
      - downsample con stride=2 su primi blocchi di stage
      - kernel sempre quadrati
    """
    def __init__(self,
                 in_ch=3,
                 stem_ch=32,
                 widths=(64, 128, 256),
                 num_classes=10,
                 head_type="invariant"):
        super().__init__()

        # ---- STEM (lifting + pool 2x) ----
        self.stem = nn.Sequential(
            LiftingConvC4(in_ch, stem_ch, k=7, stride=2),  # H/2
            nn.BatchNorm2d(stem_ch * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                # H/4
        )
        c = stem_ch

        # ---- BACKBONE: 1 blocco per stage, downsample all’inizio (stride=2) tranne primo ----
        stages = []
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2   # H/4, H/8, H/16, ...
            stages.append(C4Block(c, w, k=3, stride=stride))
            c = w
        self.backbone = nn.Sequential(*stages)

        # ---- HEAD ----
        if head_type == "invariant":
            self.head = InvariantHeadC4(c, num_classes)
        elif head_type == "equivariant":
            self.head = EquivariantHeadC4(c, out_ch=num_classes)
        else:
            raise ValueError("head_type deve essere 'invariant' o 'equivariant'")
        self.head_type = head_type

    def forward(self, x):
        # assert quadratura una volta sola, all’ingresso
        B, C, H, W = x.shape
        assert H == W, f"Atteso input quadrato, ricevuto {H}x{W}."
        x = self.stem(x)       # (B, stem_ch*4, H/4,  W/4)
        x = self.backbone(x)   # (B, c*4,        H/.., W/..)
        x = self.head(x)       # invari: (B, num_classes) | equiv: (B, num_classes*4, H', W')
        return x

    def compute_loss(self, y_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calcola la loss in base all'head.
        - Invariant head: y_pred = (B, num_classes), labels = (B,) con valori [0..num_classes-1]
        - Equivariant head (direzionale C4): out_ch=1 ⇒ y_pred = (B, 4, H', W'), labels = (B,) con valori {0,1,2,3}
        - Equivariant head (generico, out_ch>1): esegue group-pooling su C4 e poi CE su C classi.

        Ritorna: scalar loss (tensor)
        """
        if self.head_type == "invariant":
            # y_pred: (B, num_classes)
            return F.cross_entropy(y_pred, labels)

        elif self.head_type == "equivariant":
            assert y_pred.dim() == 4, "Atteso output equivariante con shape (B, C4, H, W)."
            B, C4, H, W = y_pred.shape
            assert C4 % 4 == 0, f"Numero canali {C4} non multiplo di 4."
            C = C4 // 4  # canali 'logici' per orientazione

            if C == 1:
                # Caso DIREZIONALE C4 (consigliato): out_ch=1 ⇒ 4 logit per {0°,90°,180°,270°}
                logits = F.adaptive_avg_pool2d(y_pred, 1).reshape(B, 4)  # (B,4)
                return F.cross_entropy(logits, labels)  # labels ∈ {0,1,2,3}
            else:
                # Caso GENERALE: pool sulle 4 orientazioni → classi invarianti
                y = y_pred.view(B, C, 4, H, W).mean(dim=2)                 # (B, C, H, W)
                logits = F.adaptive_avg_pool2d(y, 1).reshape(B, C)         # (B, C)
                return F.cross_entropy(logits, labels)

        else:
            raise ValueError(f"head_type sconosciuto: {self.head_type}")

    def compute_logits(self, y_pred: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = y_pred.shape
        C = C4 // 4
        y = y_pred.view(B, C, 4, H, W).mean(dim=2)                 # (B, C, H, W)
        logits = F.adaptive_avg_pool2d(y, 1).reshape(B, C)         # (B, C)
        return logits


# -----------------------
# Esempio d’uso
# -----------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)  # quadrato
    model = C4Net(in_ch=3, stem_ch=16, widths=(32, 64, 128), num_classes=5, head_type="invariant")
    y = model(x)
    print("Output shape:", y.shape)

