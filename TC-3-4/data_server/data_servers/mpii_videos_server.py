from fastapi import FastAPI
from pydantic import BaseModel
from data_servers.datasets.mpii_videos import MPIIVideos
    

mpii_videos = MPIIVideos()


async def list_mpii_videos(split: str):
    if split == "train" or split == "val" or split == "test":
        return mpii_videos.get_data_list(split)
    else:
        return None


async def list_frame_mpii_videos(split: str, video_name: str):
    if split == "train" or split == "val" or split == "test":
        return mpii_videos.get_frame_list(split, video_name)
    else:
        return None
    

async def get_frame_mpii_videos(split, video_name, file_name):
    if split == "train" or split == "val" or split == "test":
        return mpii_videos.get_frame(split, video_name, file_name)
    else:
        return None


