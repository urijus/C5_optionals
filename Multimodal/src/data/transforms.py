import torch
import torchaudio
import torch.nn.functional as F
import torchvision.transforms as transforms



# Light enough to regularize without destroyign age cues
def build_image_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.05
        ),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


class AudioTransform:
    def __init__(
        self,
        sample_rate: int = 8000,
        max_duration_sec: float = 10,
        n_mels: int = 64,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_sec)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # convert to [channel, num_samples] (now is [num_channels])
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono if needed (not in our dataset)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        # Truncate to fixed length (from 15s to 10s)
        num_samples = waveform.size(1)
        if num_samples < self.max_length:
            pad_amount = self.max_length - num_samples
            waveform = F.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :self.max_length]

        # Mel spectrogram
        mel_spec = self.mel_transform(waveform)   # [1, n_mels, time]

        # Log scale in dB
        log_mel_spec = self.db_transform(mel_spec)

        # Per-sample normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)

        return log_mel_spec