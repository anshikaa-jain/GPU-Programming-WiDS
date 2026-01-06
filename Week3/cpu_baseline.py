import numpy as np
import time

# Dimensions
H, W = 256, 256
C1, C2 = 4, 8
K = 3

# Input image
input_img = np.ones((1, H, W), dtype=np.float32)

# Weights
W1 = np.random.randn(C1, 1, K, K).astype(np.float32)
W2 = np.random.randn(C2, C1, K, K).astype(np.float32)
W3 = np.random.randn(C1, C2, K, K).astype(np.float32)

# Helper functions
def conv_relu(inp, W):
    C_out, C_in, _, _ = W.shape
    _, H, Wd = inp.shape
    out = np.zeros((C_out, H, Wd), dtype=np.float32)

    for oc in range(C_out):
        for y in range(H):
            for x in range(Wd):
                acc = 0.0
                for ic in range(C_in):
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            iy, ix = y + ky, x + kx
                            if 0 <= iy < H and 0 <= ix < Wd:
                                acc += inp[ic, iy, ix] * W[oc, ic, ky+1, kx+1]
                out[oc, y, x] = max(acc, 0.0)
    return out

def maxpool(inp):
    C, H, W = inp.shape
    out = np.zeros((C, H//2, W//2), dtype=np.float32)
    for c in range(C):
        for y in range(0, H, 2):
            for x in range(0, W, 2):
                out[c, y//2, x//2] = np.max(inp[c, y:y+2, x:x+2])
    return out

def upsample(inp):
    C, H, W = inp.shape
    out = np.zeros((C, H*2, W*2), dtype=np.float32)
    for c in range(C):
        for y in range(H):
            for x in range(W):
                out[c, y*2:y*2+2, x*2:x*2+2] = inp[c, y, x]
    return out


# CPU Timing
start = time.time()

enc1 = conv_relu(input_img, W1)
skip = enc1.copy()
enc2 = maxpool(enc1)
enc3 = conv_relu(enc2, W2)
dec1 = upsample(enc3)
dec2 = conv_relu(dec1, W3)
output = dec2 + skip

end = time.time()

print(f"[CPU] U-Net forward time: {end - start:.3f} s")
print("Output shape:", output.shape)




















# #cpu_baseline
# import torch
# import torch.nn as nn
# import time

# # -------------------------------------------------
# # Force CPU
# # -------------------------------------------------
# device = torch.device("cpu")

# # -------------------------------------------------
# # Tiny U-Net Model
# # -------------------------------------------------
# class TinyUNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, padding=1),
#             nn.ReLU()
#         )

#         self.pool = nn.MaxPool2d(2)

#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(8, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3, padding=1),
#             nn.ReLU()
#         )

#         self.up = nn.Upsample(scale_factor=2, mode='nearest')

#         self.dec1 = nn.Sequential(
#             nn.Conv2d(16, 8, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, padding=1),
#             nn.ReLU()
#         )

#         self.out = nn.Conv2d(8, 1, 1)

#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.pool(x1)
#         x3 = self.bottleneck(x2)
#         x4 = self.up(x3)
#         x5 = self.dec1(x4)
#         return self.out(x5)

# # -------------------------------------------------
# # Input: Grayscale Medical Image
# # -------------------------------------------------
# input_image = torch.randn(1, 1, 256, 256, device=device)

# model = TinyUNet().to(device)
# model.eval()

# # -------------------------------------------------
# # CPU Timing
# # -------------------------------------------------
# with torch.no_grad():
#     start = time.time()
#     output = model(input_image)
#     end = time.time()

# print(f"CPU inference time: {end - start:.4f} seconds")
# print(f"Output shape: {output.shape}")
