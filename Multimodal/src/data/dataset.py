import pickle

import pandas as pd
import torch
from scipy.io import wavfile
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset



class MultiModalDataset(Dataset):
    def __init__(
        self, 
        config, 
        split="train",
        image_transform=None,
        audio_transform=None):
        
        if split not in ["train", "valid", "test"]:
            raise ValueError("Ensure the split is: 'train', 'valid', 'test'.")
        
        self.config = config
        
        # Load labels
        self.labels_csv = pd.read_csv(self.config.data.data_dir / f"{split}_set_age_labels.csv")

        # Raw data path
        self.data_path = self.config.data.data_dir / f"{split}"

        # For loading images
        self.image_transform = image_transform

        # For loading audio
        self.audio_transform = audio_transform

    def __getitem__(self, idx):
        row = self.labels_csv.iloc[idx]
        video_name = str(self._video_name_to_stem(row["VideoName"]))
        sample_id = str(video_name)
        user_id = str(row["UserID"])
        age_label = int(row["AgeGroup"]) - 1 # 0-based classification
        gender_label = int(row["Gender"]) - 1
        ethnicity_label = int(row["Ethnicity"]) - 1 

        img_tensor, audio_tensor, text_str = None, None, None
        for modality in self.config.model.modalities:
            if modality == "image":
                img_path = self.data_path / str(age_label + 1) / f"{video_name}.jpg"
                img = Image.open(img_path).convert("RGB")
                if self.image_transform is not None:
                    img_tensor = self.image_transform(img)
                else:
                    img_tensor = transforms.PILToTensor()(img)

            elif modality == "audio":
                audio_path = self.data_path / str(age_label + 1) / f"{video_name}.wav"
                sample_rate, data = wavfile.read(audio_path)
                audio_tensor = torch.tensor(data, dtype=torch.float32)
                
                # Normalize amplitude
                if audio_tensor.abs().max() > 1:
                    audio_tensor = audio_tensor / audio_tensor.abs().max()

                if self.audio_transform is not None:
                    audio_tensor = self.audio_transform(audio_tensor, sample_rate)

            elif modality == "text":
                text_path = self.data_path / str(age_label + 1) / f"{video_name}.pkl"
                with open(text_path, "rb") as f:
                    text_str = str(pickle.load(f)).lower().strip()

            else:
                raise ValueError(f"Unsupported modality: {modality}")

        return {
            "id": sample_id,
            "user_id": user_id,
            "age": age_label,
            "gender": gender_label,
            "ethnicity": ethnicity_label,
            "image": img_tensor,
            "audio": audio_tensor,
            "text": text_str
        }

    def __len__(self):
        return len(self.labels_csv)

    def _video_name_to_stem(self, video_name: str):
        return video_name.replace(".mp4", "")


if __name__=="__main__":
    from src.config import Config
    config = Config()

    # Choose which modality to load
    config.model.modalities = ["audio"]

    #Show first value
    dataset = MultiModalDataset(config)
    audio = dataset[0]["audio"]
    print(audio)
    print(audio.shape)
