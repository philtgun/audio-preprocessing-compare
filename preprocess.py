import argparse
from pathlib import Path

import essentia.standard as ess
import librosa
import torchaudio
from tqdm import tqdm


def essentia_load(audio_file, new_sample_rate):
    return ess.MonoLoader(filename=audio_file, sampleRate=new_sample_rate)()


def torchaudio_load(audio_file, new_sample_rate):
    torchaudio.set_audio_backend('sox_io')
    waveform, sample_rate = torchaudio.load(audio_file)
    return torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)


def librosa_load(audio_file, new_sample_rate):
    waveform, sr = librosa.load(audio_file, sr=new_sample_rate)
    return waveform


def preprocess(input_dir, sample_rate):
    input_dir = Path(input_dir)

    for load in [librosa_load, essentia_load, torchaudio_load]:
        print(f'# {load.__name__}:')
        for audio_file in tqdm(input_dir.rglob('*.mp3')):
            waveform = load(str(audio_file), sample_rate)
            print(waveform.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir', help='Directory with mp3 files')
    parser.add_argument('--sample-rate', default=16000, help='Output sample rate of the audio files')
    args = parser.parse_args()
    preprocess(args.input_dir, args.sample_rate)
