"""Shared numeric constants for DiffiT.

These values were previously duplicated as inline literals across the training,
sampling and evaluation scripts. Centralizing them here removes the magic
numbers without changing any behaviour — every value is byte-for-byte identical
to the literal it replaces.
"""

# Stable-Diffusion VAE latent scaling factor. Latents are stored as
# ``z * VAE_SCALE_FACTOR``; decoding divides it back out (Rombach et al., 2022).
VAE_SCALE_FACTOR = 0.18215

# Pixel <-> [-1, 1] conversion. Encode: ``img_uint8 / PIXEL_NORM_HALF - 1``.
# Decode: ``(x + 1) * PIXEL_NORM_HALF``.
PIXEL_NORM_HALF = 127.5

# Maximum value of a uint8 image channel.
UINT8_MAX = 255

# ImageNet channel statistics used to normalize inputs to InceptionV3 when
# computing FID / IS / sFID features.
INCEPTION_MEAN = (0.485, 0.456, 0.406)
INCEPTION_STD = (0.229, 0.224, 0.225)
