from model import UNet
import numpy as np
import argparse
import torch
import os

# 1. Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type = str, default = 'result.pth')
parser.add_argument('--mixture_folder', type = str, default = 'inference/mixture')
parser.add_argument('--tar', type = str, default = 'inference/split')
args = parser.parse_args()
if not os.path.exists(args.tar):
    os.mkdir(args.tar)

# 2. Load the pre-trained model
model = UNet()
model.load(args.model_path)
model.eval()

# 3. Seperate!
with torch.no_grad():
    for idx, name in enumerate([_ for _ in sorted(os.listdir(args.mixture_folder)) if 'spec' in _]):
        mix = np.load(os.path.join(args.mixture_folder, name))
        spec_sum = None
        for i in range(mix.shape[-1] // 128):
            seg = mix[1:, i * 128:i * 128 + 128, np.newaxis]
            seg = np.asarray(seg, dtype=np.float32)
            seg = torch.from_numpy(seg).permute(2, 0, 1)
            seg = torch.unsqueeze(seg, 0)
            seg = seg.cuda()
            msk = model(seg)
            vocal_ = seg * (1 - msk)
            vocal_ = vocal_.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
            vocal_ = np.vstack((np.zeros((128)), vocal_))
            spec_sum = vocal_ if spec_sum is None else np.concatenate((spec_sum, vocal_), -1)
        np.save(os.path.join(args.tar, str(idx) + '_spec'), spec_sum)