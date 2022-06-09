import torch
import torch.nn as nn
import torch.nn.functional as functional


class MBNET_withlargeband(nn.Module):
    def __init__(self, fft_num=257, neighbor_num = 15, hidden_size=512, device='cpu') -> None:
        super().__init__()
        self.neighbor_num = neighbor_num
        self.fft_num = fft_num
        self.large_lstm = nn.LSTM(
                            input_size=fft_num,
                            num_layers=2,
                            hidden_size=hidden_size)
        self.large_band = nn.Sequential(
                            nn.Linear(in_features=hidden_size, out_features=fft_num),
                            nn.PReLU())
        self.small_lstm =  nn.LSTM(
                            input_size=2*neighbor_num+1,
                            num_layers=2,
                            hidden_size=hidden_size)
        self.small_band = nn.Sequential(
                            nn.Linear(in_features=hidden_size, out_features=2*neighbor_num+1),
                            nn.PReLU())
        self.enhance = nn.Sequential(
            nn.Linear(in_features=2*neighbor_num+1, out_features=1),
            nn.Sigmoid()
        )
        self.device = device
        self.to(device)

    def forward(self, input):
        # x ：[B, 1, F, T]
        x, y = input
        x = torch.abs(x).to(self.device)
        y = torch.abs(y).to(self.device)
        B, C, F, T = x.size()
        assert B == 1 and C == 1, "channel and batch must == 1"
        
        large_band_input = x.reshape(B, F, T) 
        
        
        spec_padded = functional.pad(x, (0, 0, self.neighbor_num, self.neighbor_num), mode="reflect")
        small_band_input = []
        for i in range(self.neighbor_num, self.neighbor_num + self.fft_num):
            a = spec_padded[:, :, i-self.neighbor_num: i + self.neighbor_num + 1, :]
            small_band_input.append(a)
        small_band_input = torch.cat(small_band_input, dim=1)
        small_band_input = small_band_input.reshape(T, F, 2*self.neighbor_num+1)

        # Large_band : [1, T, F]
        large_band_input = large_band_input.permute(0, 2, 1)
        large_band_input = self.normalize_function(large_band_input)
        large_band_out, _ = self.large_lstm(large_band_input)
        large_band_out = self.large_band(large_band_out)
        large_band_out = large_band_out.permute(1, 2, 0)
        # Small_band : [F, T, 2*N+1]
        small_band_input = small_band_input.permute(1, 0, 2)
        small_band_input = self.normalize_function(small_band_input)
        small_band_out, _ = self.small_lstm(small_band_input)
        small_band_out = self.small_band(small_band_out)
        small_band_out = small_band_out.permute(1, 0, 2)

        enhanced_small_band  = small_band_out * large_band_out
        enhanced_small_band = enhanced_small_band.permute(1, 0, 2)
        enhanced_small_band = self.enhance(enhanced_small_band)
        enhanced_small_band_mask = enhanced_small_band.reshape(B,C,F,T)
        if self.training:
            loss = functional.mse_loss(y, enhanced_small_band_mask * x)
            return loss
        return enhanced_small_band_mask * x

    def normalize_function(self, x):
        miu = torch.mean(x)
        return x / (miu+1e-6)