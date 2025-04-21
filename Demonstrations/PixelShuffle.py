import torch
import torch.nn as nn

pixel_shuffle = nn.PixelUnshuffle(2)
i = torch.randn(1, 9, 4, 4)
o = pixel_shuffle(i)
print(o.size())