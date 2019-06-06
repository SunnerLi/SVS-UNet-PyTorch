from model import UNet
from matplotlib import pyplot as plt
import torch.utils.data as Data
import numpy as np
import argparse
import librosa
import torch
import os

class SpectrogramDataset(Data.Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(os.path.join(path, 'mixture'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
            root --+-- mixture
                   |
                   +-- drums
                   |
                   +-- bass
                   |
                   +-- rest
                   |
                   +-- vocal
        """
        # import stempeg

        # wav, sr = librosa.load(self.files[idx], mono=False)

        # audio, rate = stempeg.read_stems(
        #                 filename=self.files[idx],
        #                 stem_id=4
        #             )

        # import sounddevice as sd
        # from librosa.display import waveplot
        # # waveplot(wav[1])
        # # waveplot(wav[0])
        # # plt.tight_layout()
        # # plt.show()
        # sd.play(audio, rate, blocking=True)
        # print(audio.shape)

        mix = np.load(os.path.join(self.path, 'mixture', self.files[idx]))
        voc = np.load(os.path.join(self.path, 'vocal', self.files[idx]))
        return mix, voc


"""
    Dataset: https://sigsep.github.io/datasets/musdb.html
    Ref: https://github.com/Jeongseungwoo/Singing-Voice-Separation
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
loader = Data.DataLoader(
    dataset = SpectrogramDataset('MUS-STEMS-SAMPLE/train'),
    batch_size=1, num_workers=0
)

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