"""
    mp3 -> spectrogram or spectrogram -> mp3
"""
from tqdm import tqdm
import sounddevice as sd
import numpy as np
import argparse
import librosa
import stempeg
import os

# =========================================================================================
# 1. Parse the direction and related parameters
# =========================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--src', type = str)
parser.add_argument('--tar', type = str)
parser.add_argument('--phase', type = str)
parser.add_argument('--win_size', default = 1024)
parser.add_argument('--hop_size', default = 768)
parser.add_argument('--sr', default = 44100)
parser.add_argument('--direction', default = "to_spec", help = "to_spec or to_wave")
args = parser.parse_args()

# =========================================================================================
# 2. Convert
# =========================================================================================
if args.direction == 'to_spec':
    # create the folder
    if not os.path.exists(args.tar):
        os.mkdir(args.tar)
        os.mkdir(os.path.join(args.tar, 'mixture'))
        os.mkdir(os.path.join(args.tar, 'drum'))
        os.mkdir(os.path.join(args.tar, 'bass'))
        os.mkdir(os.path.join(args.tar, 'rest'))
        os.mkdir(os.path.join(args.tar, 'vocal'))

    # load the wave, transform to spectrogram and save
    tar_folder_list = ['mixture', 'drum', 'bass', 'rest', 'vocal']
    loader = tqdm(os.listdir(args.src))
    for audio_idx, name in enumerate(loader):
        norm = None
        for i in range(5):
            audio, rate = stempeg.read_stems(filename=os.path.join(args.src, name), stem_id=i)
            audio = audio[:, 0] + audio[:, 1]
            # sd.play(audio, rate, blocking=True)
            stft = librosa.stft(audio, n_fft=args.win_size, hop_length=args.hop_size)
            spectrum, phase = librosa.magphase(stft)
            spectrogram = np.abs(spectrum).astype(np.float32)
            norm = spectrogram.max() if norm is None else norm
            spectrogram /= norm
            np.save(os.path.join(args.tar, tar_folder_list[i], str(audio_idx) + '_spec'), spectrogram)
            np.save(os.path.join(args.tar, tar_folder_list[i], str(audio_idx) + '_phase'), phase)

elif args.direction == 'to_wave':
    # create the folder
    if not os.path.exists(args.tar):
        os.mkdir(args.tar)
    
    # Load the spectrogram and merge
    loader = tqdm(os.listdir(args.src))
    for audio_idx, spec_name in enumerate(loader):
        if 'spec' in spec_name:
            # load data
            phase_name = spec_name[:-8] + 'phase.npy'
            mag = np.load(os.path.join(args.src, spec_name))
            phase = np.load(os.path.join(args.phase, phase_name))

            # resize as the same size
            length = min(phase.shape[-1], mag.shape[-1])
            mag = mag[:, :length]
            phase = phase[:, :length]

            # reconstruct the audio
            y = librosa.istft(mag*phase, win_length=args.win_size, hop_length=args.hop_size)
            file_path = os.path.join(args.tar, str(audio_idx) + '.mp3')
            librosa.output.write_wav(file_path, y, args.sr, norm=True)