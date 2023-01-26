# Work on the original DINO with PyTorch:
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
from vision_transformer import VisionTransformer
from PIL import Image
from pathlib import Path


# VI imports:
import tensorflow as tf
import json
from vit_inspect import vit_inspector as vi
from vit_inspect.summary_v2 import vi_summary

# Load the pre-trained model:
# Cach location is: '/home/stamatiad/.cache/torch/hub'
model_cached = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
# Create a version of the model that holds our attention VI modifications:
# Match model params, before load:
num_features = model_cached.embed_dim
model = VisionTransformer(embed_dim=num_features)
model.load_state_dict(model_cached.state_dict(), strict=False)

# Get some model params, required for VI:
vi.params["num_layers"] = len(model.blocks)
vi.params["num_heads"] = model.blocks[0].attn.num_heads
# The number of tokens when the attention dot product happens.
# Here tokens are the patches. Any other feature (e.g. class) is removed.
patch_size = model.patch_embed.patch_size
crop_size = 480
img_size_in_patches = crop_size // patch_size
vi.params["len_in_patches"] = img_size_in_patches
# Total patches in the image:
vi.params["num_tokens"] = img_size_in_patches ** 2

# Enable evaluation mode:
device = torch.device("cpu")
for p in model.parameters():
    p.requires_grad = False
model.eval()
model.to(device)

# Load sample images:
with open('img.png', 'rb') as f:
    img = Image.open(f)
    img = img.convert('RGB')

transform = pth_transforms.Compose([
    pth_transforms.Resize(img.size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
img = transform(img)
# make the image divisible by the patch size
w, h = img.shape[1] - img.shape[1] % patch_size, \
    img.shape[2] - img.shape[2] % patch_size
img = img[:, :w, :h].unsqueeze(0)

w_featmap = img.shape[-2] // patch_size
h_featmap = img.shape[-1] // patch_size


# Save the image into the summary:
flat_arr_rgb = tf.convert_to_tensor(
    # Make sure image's channels is the last dim:
    np.moveaxis(np.asarray(img), 1, -1)
)
with vi.writer.as_default():
    step = 0
    batch_id = 0
    vi.params["step"] = 0
    vi.params["batch_id"] = batch_id
    vi_summary(
        f"b{batch_id}",
        flat_arr_rgb,
        step=step,
        description=json.dumps(vi.params)
    )
    vi.writer.flush()

# Use the VI context manager to get attention maps of each layer and head:
with vi.enable_vi():
    attentions = model.get_last_selfattention(img.to(device))



