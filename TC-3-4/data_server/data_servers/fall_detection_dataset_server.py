from fastapi import FastAPI
from pydantic import BaseModel
from data_servers.datasets.fall_detection_dataset import FallDetectionDataset
    

fall_detection_dataset = FallDetectionDataset()


async def list_fall_detection_dataset(split: str):
    if split == "train" or split == "val" or split == "test":
        return fall_detection_dataset.get_data_list(split)
    else:
        return None


async def list_frame_fall_detection_dataset(split: str, video_name: str):
    if split == "train" or split == "val" or split == "test":
        return fall_detection_dataset.get_frame_list(split, video_name)
    else:
        return None
    

async def get_frame_fall_detection_dataset(split, video_name, file_name):
    if split == "train" or split == "val" or split == "test":
        return fall_detection_dataset.get_frame(split, video_name, file_name)
    else:
        return None


