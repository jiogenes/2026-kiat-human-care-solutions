from fastapi import FastAPI
from pydantic import BaseModel
from data_servers.datasets.multiple_camera_fall_dataset import MultipleCameraFallDataset
    

multiple_camera_fall_dataset = MultipleCameraFallDataset()


async def list_multiple_camera_fall_dataset(split: str):
    if split == "train" or split == "val" or split == "test":
        return multiple_camera_fall_dataset.get_data_list(split)
    else:
        return None


async def list_frame_multiple_camera_fall_dataset(split: str, video_name: str):
    if split == "train" or split == "val" or split == "test":
        return multiple_camera_fall_dataset.get_frame_list(split, video_name)
    else:
        return None
    

async def get_frame_multiple_camera_fall_dataset(split, video_name, file_name):
    if split == "train" or split == "val" or split == "test":
        return multiple_camera_fall_dataset.get_frame(split, video_name, file_name)
    else:
        return None


