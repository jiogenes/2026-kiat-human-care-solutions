import data_servers
import data_servers.datasets
import pathlib
import os
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


@app.get("/")
async def hello_world():
    return "hello, world"


@app.get("/dataset/")
async def list_dataset():
    dataset_list = []

    for path in pathlib.Path("./data_servers/datasets").glob("*.py"):
        dataset_name = os.path.splitext(path.name)[0]
        dataset_list.append({
            "name": dataset_name,
            "type": getattr(getattr(data_servers.datasets, dataset_name), "DATA_TYPE")
        })

    return dataset_list


@app.get("/{dataset}/list_data/{split}/")
async def list_dataset_split(dataset:str, split: str):
    list_dataset_func = getattr(data_servers, f"list_{dataset}")
    return await list_dataset_func(split)


@app.get("/{dataset}/list_frames/{split}/{video_name}/")
async def list_dataset_video(dataset:str, split: str, video_name: str):
    data_type = getattr(getattr(data_servers.datasets, dataset), "DATA_TYPE")
    if data_type == "video":
        list_frame_dataset_func = getattr(data_servers, f"list_frame_{dataset}")
        return await list_frame_dataset_func(split, video_name)
    else:
        return None


@app.get("/{dataset}/get_frame/{split}/{video_name}/{file_name}")
async def get_frame(dataset:str, split: str, video_name: str, file_name: str):
    data_type = getattr(getattr(data_servers.datasets, dataset), "DATA_TYPE")
    if data_type == "video":
        get_frame_func = getattr(data_servers, f"get_frame_{dataset}")
        return await get_frame_func(split, video_name, file_name)
    else:
        return None


@app.get("/{dataset}/get_sound/{split}/{file_name}")
async def get_sound(dataset:str, split: str, file_name: str):
    data_type = getattr(getattr(data_servers.datasets, dataset), "DATA_TYPE")
    if data_type == "sound":
        get_sound_func = getattr(data_servers, f"get_sound_{dataset}")
        return await get_sound_func(split, file_name)
    else:
        return None


@app.get("/{dataset}/get_signal/{split}/")
async def get_signal(dataset:str, split: str):
    data_type = getattr(getattr(data_servers.datasets, dataset), "DATA_TYPE")
    if data_type == "signal":
        get_signal_func = getattr(data_servers, f"get_signal_{dataset}")
        return await get_signal_func(split)
    else:
        return None
