import torchaudio
from torch.utils.data import Dataset
import os
import torch


class MyDataset(Dataset):
    def __init__(self, audio_path = "E:\\dataset\\voicebank", is_training=True):
        super(MyDataset, self).__init__()
        self.state = None
        self.clean_list = []
        self.noisy_list = []
        if is_training:
            self.state = "train"
            self.path = os.path.join(audio_path, self.state)
        else:
            self.state = "test"
            self.path = os.path.join(audio_path, self.state)

        for root, dirs, files in os.walk(self.path):
            if len(files) != 0:
                if "clean" in root:
                    self.clean_list = [root + "/" + file for file in files]
                elif "noisy" in root:
                    self.noisy_list = [root + "/" + file for file in files]
        self.clean_list = sorted(self.clean_list)
        self.noisy_list = sorted(self.noisy_list)

    def __getitem__(self, index):
        '''

        :param index:
        :return: cIRM :shape [n_fft, frames, 2]
        '''
        clean_path = self.clean_list[index]
        noisy_path = self.noisy_list[index]

        label, _ = torchaudio.load(clean_path)
        noisy_sig, _ = torchaudio.load(noisy_path)
        noisy_sig = torch.stft(noisy_sig, n_fft=512, hop_length=256, return_complex=True, window=torch.hann_window(512))
        label = torch.stft(label, n_fft=512, hop_length=256, return_complex=True, window=torch.hann_window(512))
        return noisy_sig, label

    def __len__(self):
        return len(self.noisy_list)