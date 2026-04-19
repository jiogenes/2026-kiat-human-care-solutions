import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "utils"))

import numpy as np
import torch, einops
# import foolbox as fb
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import InceptionScore
from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from augmentation import VideoAugmenter
from preprocessors.data.dataset import VideoDataset
from sqrtm import sqrtm
from tqdm import tqdm


device = torch.device("cuda:1")
batch_size = 8


def compute_fvd(fake_f, real_f):
    mean_fake = fake_f.mean(dim=0)
    mean_real = real_f.mean(dim=0)

    cov_fake = fake_f.t().mm(fake_f) / fake_f.size(0)
    cov_real = real_f.t().mm(real_f) / real_f.size(0)
    m = ((mean_fake - mean_real)**2).sum()
    s = (cov_fake + cov_real - 2 * sqrtm(cov_fake.mm(cov_real))).trace()
    fvd = m + s
    return fvd


def to_uint8(tensor):
    tensor = tensor*0.5 + 0.5
    tensor = (tensor*255).type(torch.uint8)
    return tensor


def run():
    dataset = VideoDataset("/data/kiat/mpii_videos_flatten", image_shape=(224, 224))
    print(f"Length of dataset: {len(dataset)}")

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

    inception_scorer = InceptionScore(splits=10).to(device)

    i3d = load_i3d_pretrained(device)
    for param in i3d.parameters():
        param.requires_grad_(False)

    real_features = []
    fake_features = []

    N = len(dataset)
    num_batches = int(np.ceil(N / batch_size))
    for b in range(num_batches):
        start_index = b*batch_size
        end_index = min(N, (b + 1)*batch_size)

        real_video = map(lambda i: dataset[i]["video"], range(start_index, end_index))
        min_length = min(real_video, key=lambda t: t.size(0)).size(0)
        real_video = list(map(lambda t: t[:min_length]))
        real_video = torch.stack(real_video, dim=0)

        N, T, C, H, W = real_video.size()
        fake_video = augmenter.apply(real_video)

        real_video = einops.rearrange(real_video, "n t c h w -> n c t h w").to(device)
        fake_video = einops.rearrange(fake_video, "n t c h w -> n c t h w").to(device)

        with torch.no_grad():
            real_f = i3d(real_video, return_features=True)
            fake_f = i3d(fake_video, return_features=True)

        real_features.append(real_f.cpu().numpy())
        fake_features.append(fake_f.cpu().numpy())

        with torch.no_grad():
            fake_video_uint8 = to_uint8(fake_video)
            fake_video_uint8 = einops.rearrange(fake_video_uint8, "n c t h w -> (n t) c h w")
            inception_scorer.update(fake_video_uint8)

        if b == 100: break

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    fvd = frechet_distance(fake_f.cpu().numpy(), real_f.cpu().numpy())
    inception_score = inception_scorer.compute()[0]
    print(f"FVD: {fvd}")
    print(f"IS: {inception_score}")


if __name__ == "__main__":
    run()
