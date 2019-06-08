"""
    This script will convert the audio to spectrogram, or convert spectrogram to audio
    As in former direction, the five folder will be created.
    We follow the format of MuseDB18 which contains 5 different tracks. 

            root --+-- mixture
                   |
                   +-- drum
                   |
                   +-- bass
                   |
                   +-- rest
                   |
                   +-- vocal

    For the latter direction, you should assign both the magnitude and phase folder
    The site of MuseDB18 is here: https://sigsep.github.io/datasets/musdb.html

    @Author: SunnerLi
"""
from utils import num2str
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
"""
                                    Parameter Explain
    --------------------------------------------------------------------------------------------
        --src               The source folder where the song you want to split are locate in
                            (Or the magnitude you want to reconstruct)
        --tar               The target folder where the spectrogram will store in
                            (Or the folder where you want to store the reconstruct wave)
        --phase             The folder of phase. You only need to assign in to_wave direction
        --win_size          The window size of STFT. The original paper use 1024
        --hop_size          The hop size of window. The original paper use 768
        --sr                The sampling rate. Default is 44100
        --direction         The direction you want to deal with. 'to_spec' or 'to_wave'
    --------------------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--src'         , type = str)
parser.add_argument('--tar'         , type = str)
parser.add_argument('--phase'       , type = str, default = '-1')
parser.add_argument('--win_size'    , default = 1024)
parser.add_argument('--hop_size'    , default = 768)
parser.add_argument('--sr'          , default = 44100)
parser.add_argument('--direction'   , default = "to_spec")
args = parser.parse_args()
if args.direction == 'to_wave' and args.phase == '-1':
    raise Exception("You need to assign the phase parameter in to_wave direction!")

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
    loader = tqdm(sorted(os.listdir(args.src)))
    for audio_idx, name in enumerate(loader):
        norm = None
        for i in range(5):
            try:
                audio, rate = stempeg.read_stems(filename=os.path.join(args.src, name), stem_id=i)
                audio = audio[:, 0] + audio[:, 1]
                # sd.play(audio, rate, blocking=True)
                stft = librosa.stft(audio, n_fft=args.win_size, hop_length=args.hop_size)
                spectrum, phase = librosa.magphase(stft)
                spectrogram = np.abs(spectrum).astype(np.float32)
                norm = spectrogram.max() if norm is None else norm
                spectrogram /= norm
                np.save(os.path.join(args.tar, tar_folder_list[i], num2str(audio_idx) + '_' + name[:-10] + '_spec'), spectrogram)
                np.save(os.path.join(args.tar, tar_folder_list[i], num2str(audio_idx) + '_' + name[:-10] + '_phase'), phase)
            except IndexError:
                print("[Warning] The song {} cannot capture the {} track".format(name, tar_folder_list[i]))

elif args.direction == 'to_wave':
    # create the folder
    if not os.path.exists(args.tar):
        os.mkdir(args.tar)
    
    # Load the spectrogram and merge
    loader = tqdm(sorted(os.listdir(args.src)))
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

            # De-normalization
            spectrogram = mag*phase
            mix_spec = np.load(os.path.join(args.phase, spec_name))
            spectrogram *= mix_spec.max()

            # reconstruct the audio
            y = librosa.istft(spectrogram, win_length=args.win_size, hop_length=args.hop_size)
            file_path = os.path.join(args.tar, str(audio_idx) + '.mp3')
            librosa.output.write_wav(file_path, y, int(args.sr), norm=True)

else:
    raise Exception("Unknown direction {}. Please assign one of them [to_spec, to_wave]".format(args.direction))