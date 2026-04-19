import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from typing import *


class VideoAugmenter():

    def __init__(self, color_jitter: bool = True,
                       brightness: List[float] = [0.8, 1.2],
                       contrast: List[float] = [0.8, 1.2],
                       saturation: List[float] = [0.8, 1.2],
                       hue: List[float] = [-0.02, 0.02],
                       affine: bool = True, 
                       rotation: List[float] = [-0.1*np.pi, 0.1*np.pi],
                       translation: List[float] = [-0.15, 0.15],
                       scale: List[float] = [0.85, 1.15],
                       gaussian_noise: bool = True,
                       std: float = 0.01):
        
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.affine = affine
        self.rotation = rotation
        self.translation = translation
        self.scale = scale

        self.gaussian_noise = gaussian_noise
        self.std = std

    def apply(self, video: torch.Tensor):
        """
        Arguments:
            video: (T, C, H, W) or (B, T, C, H, W). B and T is number of batches and number of frames. value shoud have range of [-1, 1]
        """

        video = video*0.5 + 0.5

        if len(video.shape) == 5:
            augmented_video = list(map(self._apply, video))
            augmented_video = torch.stack(augmented_video, dim=0)
        else:
            augmented_video = self._apply(video)

        augmented_video = (augmented_video - 0.5)/0.5
        augmented_video.clamp_(-1, 1)
        return augmented_video
    
    def _apply(self, video: torch.Tensor):
        """
        Arguments:
            video: (T, C, H, W). T is number of batches and number of frames
        """

        augmented_video = video

        if self.color_jitter is True:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self._get_random_color_jitter_parameters()

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    augmented_video = TF.adjust_brightness(augmented_video, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    augmented_video = TF.adjust_contrast(augmented_video, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    augmented_video = TF.adjust_saturation(augmented_video, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    augmented_video = TF.adjust_hue(augmented_video, hue_factor)

        if self.gaussian_noise is True:
            augmented_video = augmented_video + torch.randn_like(augmented_video)*self.std

        if self.affine is True:
            grid, affine_mat = self._get_random_affine_parameters(video)
            augmented_video = F.grid_sample(augmented_video, grid, mode="bilinear", align_corners=False)

        return augmented_video

    # def _get_random_affine_parameters(self, video):
    #     # rotation_factor = [-0.25*np.pi, 0.25*np.pi]
    #     rotation_factor = [-10, 10]
    #     offset_factor = [0.15, 0.15]
    #     scale_factor = [0.85, 1.15]

    #     angle, translations, scale, shear = T.RandomAffine.get_params(
    #         rotation_factor,
    #         offset_factor,
    #         scale_factor,
    #         None,
    #         video.size()[-2:]
    #     )

    #     return angle, translations, scale, shear

    def _get_random_color_jitter_parameters(self):
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue
        )

        return fn_idx, b, c, s, h

    def _get_random_affine_parameters(self, video):
        angle = np.random.uniform(*self.rotation)
        scale = np.random.uniform(*self.scale)
        translation = np.random.uniform(*self.translation, size=2)

        affine_mat = torch.tensor([
            [scale*np.cos(angle), -scale*np.sin(angle), translation[0]],
            [scale*np.sin(angle), scale*np.cos(angle), translation[1]],
        ])
        affine_mat = affine_mat[None, :, :].expand(video.size(0), 2, 3).float()

        grid = F.affine_grid(affine_mat, size=video.size(), align_corners=False)
        return grid, affine_mat
    
    # def _apply(self, video: torch.Tensor):
    #     """
    #     Arguments:
    #         video: (T, C, H, W). T is number of batches and number of frames
    #     """

    #     augmented_frames = [video[t] for t in range(len(video))]

    #     if self.color_jitter is True:
    #         fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self._get_random_color_jitter_parameters()

    #         for t in range(len(augmented_frames)):
    #             frame = augmented_frames[t]
    #             for fn_id in fn_idx:
    #                 if fn_id == 0 and brightness_factor is not None:
    #                     frame = TF.adjust_brightness(frame, brightness_factor)
    #                 elif fn_id == 1 and contrast_factor is not None:
    #                     frame = TF.adjust_contrast(frame, contrast_factor)
    #                 elif fn_id == 2 and saturation_factor is not None:
    #                     frame = TF.adjust_saturation(frame, saturation_factor)
    #                 elif fn_id == 3 and hue_factor is not None:
    #                     frame = TF.adjust_hue(frame, hue_factor)

    #             augmented_frames[t] = frame

    #     augmented_video = torch.stack(augmented_frames, dim=0)

    #     if self.affine is True:
    #         grid, affine_mat = self._get_random_affine_parameters(video)
    #         augmented_video = F.grid_sample(augmented_video, grid, mode="bicubic")

    #     #     angle, translations, scale, shear = self._get_random_affine_parameters(video)
    #     #     for t in range(len(augmented_frames)):
    #     #         augmented_frames[t] = TF.affine(augmented_frames[t], angle, translations, scale, shear, interpolation=TF.InterpolationMode.BILINEAR)

    #     return augmented_video
