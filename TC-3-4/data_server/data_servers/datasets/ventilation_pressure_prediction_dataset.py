import pathlib
import cv2
from fastapi.responses import FileResponse

DATA_TYPE = "signal"

class VentilationPressurePredictionDataset():
    DATA_ROOT = "/data/kiat/ventilation_pressure_prediction/"

    def __init__(self):
        pass

    def get_split_info(self):
        return [
            "train", "test"
        ]
    
    def get_signal(self, split):
        data_file = f"{self.DATA_ROOT}/{split}.csv"
        return FileResponse(data_file)
