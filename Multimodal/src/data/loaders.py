from torch.utils.data import DataLoader

from src.data.dataset import MultiModalDataset
from src.data.transforms import build_image_transforms, AudioTransform
from src.data.collate import multimodal_collate_fn
from src.data.sampler import build_weighted_sampler


def get_datasets(config):
    img_train_transform, img_eval_transform = build_image_transforms(config.data.image_size)
    audio_transform = AudioTransform(
        sample_rate=config.data.sample_rate,
        n_mels=config.data.n_mels,
    )
    train_dataset = MultiModalDataset(
        config, split="train", 
        image_transform=img_train_transform,
        audio_transform=audio_transform)
    
    valid_dataset = MultiModalDataset(
        config, 
        split="valid",
        image_transform=img_eval_transform,
        audio_transform=audio_transform)
    test_dataset = MultiModalDataset(
        config, 
        split="test",
        image_transform=img_eval_transform,
        audio_transform=audio_transform)
    
    return train_dataset, valid_dataset, test_dataset

def get_dataloaders(config):
    train_dataset, valid_dataset, test_dataset = get_datasets(config)

    # Sampler for train only
    train_sampler = None
    shuffle_train = True

    if config.train.use_weighted_sampler:
        train_labels = train_dataset.labels_csv["AgeGroup"].to_list()
        train_labels = [train_label - 1 for train_label in train_labels]
        train_sampler, class_counts, class_weights = build_weighted_sampler(train_labels)
        shuffle_train = False
    else:
        class_counts = None
        class_weights = None

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, class_counts, class_weights


if __name__=="__main__":
    from src.config import Config
    config = Config()
    config.model.modalities = ["image", "audio", "text"]

    train_loader, valid_loader, test_loader, class_counts, class_weights = get_dataloaders(config)

    batch = next(iter(train_loader))

    print(batch.keys())
    print("ages:", batch["age"].shape)
    print("gender:", batch["gender"].shape)
    print("ethnicity:", batch["ethnicity"].shape)

    if batch["image"] is not None:
        print("image:", batch["image"].shape)

    if batch["audio"] is not None:
        print("audio:", batch["audio"].shape)

    if batch["text"] is not None:
        print("num texts:", len(batch["text"]))
        print("first text:", batch["text"][0][:120])

    print("class counts:", class_counts)
    print("class weights:", class_weights)