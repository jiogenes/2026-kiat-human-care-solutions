import pathlib
import cv2
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

DATA_TYPE = "sound"

class RespiratorySoundDataset():
    DATA_ROOT = "/data/kiat/respiratory_sound_dataset/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files"

    def __init__(self):
        pass

    def get_split_info(self):
        return [
            "train"
        ]
    
    def get_data_list(self, split):
        data_dir = f"{self.DATA_ROOT}"
        data_list = []
        
        for path in pathlib.Path(data_dir).glob("*"):
            if path.name.endswith(".wav"):
                data_list.append(path.name)

        return data_list
    
    def get_sound(self, split, file_name):
        data_path = f"{self.DATA_ROOT}/{file_name}"
        return FileResponse(data_path)
