import numpy as np
import torch, einops
from torch.nn import functional as F
from torch.utils.data import DataLoader
try:
    from torchmetrics.image.inception import InceptionScore
except:
    from torchmetrics.image import InceptionScore
from data_servers.preprocessors.video_preprocessors.augmentation import VideoAugmenter
from data_servers.preprocessors.data.dataset import VideoDataset, ImageDataset
from tqdm import tqdm



device = torch.device("cuda")
batch_size = 8


def to_uint8(tensor):
    tensor = tensor*0.5 + 0.5
    tensor = (tensor*255).type(torch.uint8)
    return tensor


def my_collate(data_list):
    data_list = list(map(lambda data_item: data_item["video"], data_list))
    min_length = min(data_list, key=lambda t: t.size(0)).size(0)
    batch = list(map(lambda t: t[:min_length], data_list))
    batch = torch.stack(batch, dim=0)
    return batch


def run():
    video_dataset = VideoDataset("/data/moon/kiat/mpii_videos_flatten", image_shape=(224, 224))
    image_dataset = ImageDataset("/data/moon/kiat/imagenet/train", image_shape=(224, 224))
    print(len(video_dataset), len(image_dataset))

    print(f"Length of dataset: {len(video_dataset) + len(image_dataset)}")

    video_dloader = DataLoader(video_dataset, batch_size=batch_size, num_workers=8, collate_fn=my_collate)
    image_dloader = DataLoader(image_dataset, batch_size=128, num_workers=8)

    augmenter = VideoAugmenter(
        gaussian_noise=False, 
        std=0.01,
        affine=False,
        rotation=[-np.pi*0.05, np.pi*0.05],
        translation=[-0.01, 0.01],
        scale=[0.95, 1.05],
        color_jitter=True,
        brightness=[0.9, 1.1],
        contrast=[0.9, 1.1],
        saturation=[0.9, 1.1],
        hue=[-0.01, 0.01]
    )

    inception_scorer = InceptionScore(splits=1).to(device)

    for b, real_video in enumerate(tqdm(video_dloader)):
        real_video = real_video.to(device)

        T = real_video.size(1)
        real_video = einops.rearrange(real_video, "n t c h w -> (n t) c h w")
        fake_video = augmenter.apply(real_video)

        with torch.no_grad():
            fake_video_uint8 = to_uint8(fake_video)
            inception_scorer.update(fake_video_uint8)

    for b, batch in enumerate(tqdm(image_dloader)):
        image = batch["image"].to(device)
        image = augmenter.apply(image)

        with torch.no_grad():
            image_uint8 = to_uint8(image)
            inception_scorer.update(image_uint8)

    inception_score = inception_scorer.compute()[0]
    print(f"IS: {inception_score}")


if __name__ == "__main__":
    run()
