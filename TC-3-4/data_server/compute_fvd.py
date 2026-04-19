import numpy as np
import torch, einops, time
# import foolbox as fb
# from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_servers.utils.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
from data_servers.preprocessors.video_preprocessors.augmentation import VideoAugmenter
from data_servers.preprocessors.data.dataset import VideoDataset
from data_servers.utils.sqrtm import sqrtm
from tqdm import tqdm




device = torch.device("cuda")
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


def my_collate(data_list):
    data_list = list(map(lambda data_item: data_item["video"], data_list))
    min_length = min(data_list, key=lambda t: t.size(0)).size(0)
    batch = list(map(lambda t: t[:min_length], data_list))
    batch = torch.stack(batch, dim=0)
    return batch


def run():
    dataset = VideoDataset("/data/moon/kiat/mpii_videos_flatten", image_shape=(224, 224))
    print(f"Length of dataset: {len(dataset)}")

    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=my_collate)

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

    i3d = load_i3d_pretrained(device)
    for param in i3d.parameters():
        param.requires_grad_(False)

    real_features = []
    fake_features = []

    for b, real_video in enumerate(tqdm(dloader)):
        real_video = real_video.to(device)

        T = real_video.size(1)
        real_video = einops.rearrange(real_video, "n t c h w -> (n t) c h w")
        fake_video = augmenter.apply(real_video)

        real_video = einops.rearrange(real_video, "(n t) c h w -> n c t h w", t=T)
        fake_video = einops.rearrange(fake_video, "(n t) c h w -> n c t h w", t=T)

        with torch.no_grad():
            real_f = i3d(real_video, return_features=True)
            fake_f = i3d(fake_video, return_features=True)

        real_features.append(real_f.cpu().numpy())
        fake_features.append(fake_f.cpu().numpy())

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    
    fvd = frechet_distance(fake_f.cpu().numpy(), real_f.cpu().numpy())
    print(f"FVD: {fvd}")


if __name__ == "__main__":
    run()
