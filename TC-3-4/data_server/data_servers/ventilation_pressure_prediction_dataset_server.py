from fastapi import FastAPI
from pydantic import BaseModel
from data_servers.datasets.ventilation_pressure_prediction_dataset import VentilationPressurePredictionDataset
    

ventilation_pressure_prediction_dataset = VentilationPressurePredictionDataset()
    

async def get_signal_ventilation_pressure_prediction_dataset(split):
    if split == "train" or split == "val" or split == "test":
        return ventilation_pressure_prediction_dataset.get_signal(split)
    else:
        return None


