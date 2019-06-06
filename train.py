from model import UNet
import torch.utils.data as Data
import argparse
import torch
import os

class SpectrogramLoader(Data.DataLoader):
    pass


"""
    Dataset: https://sigsep.github.io/datasets/musdb.html
"""

# 1. Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--mixture', type = str, default = './data/mixture')
parser.add_argument('--vocals', type = str, default = './data/vocals')
parser.add_argument('--load_path', type = str, default = 'result.pth')
parser.add_argument('--save_path', type = str, default = 'result.pth')
parser.add_argument('--epoch', type = int, default = 1)
args = parser.parse_args()

# 2. Create the data loader
loader = SpectrogramLoader()

# 3. Load the pre-trained model
model = UNet()
model.load(args.load_path)

# 4. Train!
for ep in range(args.epoch):
    bar = tqdm_table(loader)
    for i, (mix, voc) in enumerate(bar):
        model.backward(mix, voc)
        bar.set_table_info(model.getLoss(normalize = False))
        if i == len(bar) - 1:
            bar.set_table_info(model.getLoss(normalize = True))
    model.save(args.save_path)
print("Finish training!")