import torch
from dataset import MyDataset
from torch.utils.data import DataLoader
import soundfile
from model import MBNET_withoutlargeband


if __name__ == "__main__":
    checkpoint = torch.load('epoch_0.pth',map_location=torch.device('cpu'))
    dataset = MyDataset(is_training=False)
    model_state_dict = checkpoint['model']
    model = MBNET_withoutlargeband()
    model.load_state_dict(model_state_dict)
    model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for input in dataloader:
        x, y = input
        pred = model((x, y))
        phase = torch.angle(x)
        # phase = torch.randn_like(phase)
        pred = torch.complex(pred * torch.cos(phase), pred * torch.sin(phase))
        pred = torch.squeeze(pred, dim=0)
        pred = torch.istft(pred, n_fft=512, hop_length=256, window=torch.hann_window(512))
        pred = pred.detach().squeeze(0).cpu().numpy()
        soundfile.write("pred.wav", pred, 16000)
        break