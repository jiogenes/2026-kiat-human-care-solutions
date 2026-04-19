import pathlib
import cv2
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

DATA_TYPE = "video"

class MultipleCameraFallDataset():
    DATA_ROOT = "/data/kiat/multiple_camera_fall_dataset"

    def __init__(self):
        pass

    def get_split_info(self):
        return [
            "train"
        ]
    
    def get_data_list(self, split):
        data_dir = f"{self.DATA_ROOT}/dataset"
        data_list = []
        
        for path in pathlib.Path(data_dir).glob("*/*"):
            if path.is_dir():
                data_list.append(f"{path.parent.name}-{path.name}")

        return data_list
    
    def get_frame_list(self, split, video_name):
        data_dir = f"{self.DATA_ROOT}/dataset/{video_name.split('-')[0]}/{video_name.split('-')[1]}/"
        print(data_dir)
        data_list = []

        for path in pathlib.Path(data_dir).glob("*.*"):
            if path.name.endswith(".png") or path.name.endswith(".jpg"):
                data_list.append(path.name)

        return data_list
    
    def get_frame(self, split, video_name, file_name):
        data_path = f"{self.DATA_ROOT}/dataset/{video_name.split('-')[0]}/{video_name.split('-')[1]}/{file_name}"
        return FileResponse(data_path)
