import pathlib
import cv2
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

DATA_TYPE = "video"

class FallDetectionDataset():
    DATA_ROOT = "/data/kiat/fall_detection_dataset"

    def __init__(self):
        pass

    def get_split_info(self):
        return [
            "train", "test", "validate"
        ]
    
    def get_data_list(self, split):
        data_dir = f"{self.DATA_ROOT}/{split}"
        data_list = []
        
        for path in pathlib.Path(data_dir).glob("*"):
            if path.is_dir():
                data_list.append(path.name)

        return data_list
    
    def get_frame_list(self, split, video_name):
        data_dir = f"{self.DATA_ROOT}/{split}/{video_name}/rgb/"
        data_list = []

        for path in pathlib.Path(data_dir).glob("*.*"):
            if path.name.endswith(".png") or path.name.endswith(".jpg"):
                data_list.append(path.name)

        return data_list
    
    def get_frame(self, split, video_name, file_name):
        data_path = f"{self.DATA_ROOT}/{split}/{video_name}/rgb/{file_name}"
        return FileResponse(data_path)
