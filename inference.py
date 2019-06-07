from utils import num2str
from model import UNet
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os

"""
    This script defines the inference procedure of SVS-UNet

    @Author: SunnerLi
"""
# =========================================================================================
# 1. Parse the direction and related parameters
# =========================================================================================
"""
                                    Parameter Explain
    --------------------------------------------------------------------------------------------
        --model_path        The path of pre-trained model
        --mixture_folder    The root of the testing folder. You can generate via data.py 
        --tar               The folder where you want to save the splited magnitude in
    --------------------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model_path'      , type = str, default = 'result.pth')
parser.add_argument('--mixture_folder'  , type = str, default = 'inference/mixture')
parser.add_argument('--tar'             , type = str, default = 'inference/split')
args = parser.parse_args()
if not os.path.exists(args.tar):
    os.mkdir(args.tar)

# =========================================================================================
# 2. Separate the singing voice for the song
# =========================================================================================
# Load the pre-trained model
model = UNet()
model.load(args.model_path)
model.eval()

# Seperate!
with torch.no_grad():
    bar = tqdm([_ for _ in sorted(os.listdir(args.mixture_folder)) if 'spec' in _])
    for idx, name in enumerate(bar):
        if idx > 5:
            break
        mix = np.load(os.path.join(args.mixture_folder, name))
        spec_sum = None
        for i in range(mix.shape[-1] // 128):
            # Get the fixed size of segment
            seg = mix[1:, i * 128:i * 128 + 128, np.newaxis]
            seg = np.asarray(seg, dtype=np.float32)
            seg = torch.from_numpy(seg).permute(2, 0, 1)
            seg = torch.unsqueeze(seg, 0)
            seg = seg.cuda()

            # generate mask
            msk = model(seg)

            # split the voice
            vocal_ = seg * (1 - msk)
            # vocal_ = seg * msk

            # accumulate the segment until the whole song is finished
            vocal_ = vocal_.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
            vocal_ = np.vstack((np.zeros((128)), vocal_))
            spec_sum = vocal_ if spec_sum is None else np.concatenate((spec_sum, vocal_), -1)
        np.save(os.path.join(args.tar, num2str(idx) + '_' + name[5:-9] + '_spec'), spec_sum)