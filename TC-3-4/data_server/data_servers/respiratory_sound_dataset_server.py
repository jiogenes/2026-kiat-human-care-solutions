from fastapi import FastAPI
from pydantic import BaseModel
from data_servers.datasets.respiratory_sound_dataset import RespiratorySoundDataset
    

respiratory_sound_dataset = RespiratorySoundDataset()


async def list_respiratory_sound_dataset(split: str):
    if split == "train" or split == "val" or split == "test":
        return respiratory_sound_dataset.get_data_list(split)
    else:
        return None
    

async def get_sound_respiratory_sound_dataset(split, file_name):
    if split == "train" or split == "val" or split == "test":
        return respiratory_sound_dataset.get_sound(split, file_name)
    else:
        return None


