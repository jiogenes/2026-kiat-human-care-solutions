import numpy as np
import torch, cv2, pathlib
from torch.utils.data import Dataset
from ..video_preprocessors.augmentation import VideoAugmenter


class VideoDataset(Dataset):

    def __init__(self, data_dir, image_shape=(224, 224)):
        self.video_list = self._load_data_list(data_dir)
        self.image_shape = image_shape

    def _load_data_list(self, data_dir):
        video_list = []

        video_path_list = list(pathlib.Path(data_dir).glob("*"))
        video_path_list.sort()

        for video_dir in video_path_list:
            video_id = video_dir.name
            frame_list = []
            frame_path_list = list(video_dir.glob("*.jpg"))
            frame_path_list.sort()
            for frame in frame_path_list:
                frame_list.append(frame)

            if len(frame_list) > 0:
                video_list.append((video_id, frame_list))

        return video_list
    
    def __len__(self):
        return len(self.video_list)
    
    def _load_frame(self, frame_path):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = frame.shape[:2]
        frame = cv2.resize(frame, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_LINEAR)

        frame = (frame.astype(np.float32) - 127.5)/127.5
        frame = torch.tensor(frame, dtype=torch.float32)
        frame = frame.permute(2, 0, 1)
        return frame, [H, W]

    def _load_video(self, frame_path_list):
        frames = []
        orig_shapes = []
        for frame_path in frame_path_list:
            frame, orig_shape = self._load_frame(frame_path)
            frames.append(frame)
            orig_shapes.append(orig_shape)

        video = torch.stack(frames, dim=0)
        return video, orig_shapes

    def __getitem__(self, index):
        vid, frame_path_list = self.video_list[index]
        video, orig_shape = self._load_video(frame_path_list)

        return {
            "video_id": vid,
            "video": video,
            "orig_shape": orig_shape,
            "frame_name": [p.name for p in frame_path_list]
        }


class MPIIDataset(Dataset):

    def __init__(self, image_shape=(224, 224)):
        self.video_list = self._load_data_list()
        self.image_shape = image_shape

    def _load_data_list(self):
        video_list = []

        for video_dir in pathlib.Path("/data/kiat/mpii_videos").glob("*/*"):
            video_id = video_dir.name
            frame_list = []
            for frame in video_dir.glob("*.jpg"):
                frame_list.append(frame)

            frame_list.sort()
            if len(frame_list) > 0:
                video_list.append((video_id, frame_list))

        video_list.sort(key=lambda t: t[0])
        return video_list
    
    def __len__(self):
        return len(self.video_list)
    
    def _load_frame(self, frame_path):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        H, W = frame.shape[:2]
        frame = cv2.resize(frame, dsize=(self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = (frame.astype(np.float32) - 127.5)/127.5
        frame = torch.tensor(frame, dtype=torch.float32)
        frame = frame.permute(2, 0, 1)
        return frame, [H, W]

    def _load_video(self, frame_path_list):
        frames = []
        orig_shapes = []
        for frame_path in frame_path_list:
            frame, orig_shape = self._load_frame(frame_path)
            frames.append(frame)
            orig_shapes.append(orig_shape)

        video = torch.stack(frames, dim=0)
        return video, orig_shapes

    def __getitem__(self, index):
        vid, frame_path_list = self.video_list[index]
        video, orig_shape = self._load_video(frame_path_list)
        return {
            "video_id": vid,
            "video": video,
            "orig_shape": orig_shape,
            "frame_name": [p.name for p in frame_path_list]
        }
    

class ImageDataset(Dataset):

    def __init__(self, data_dir, image_shape=(224, 224)):
        self.data_list = self._load_data_list(data_dir)
        self.image_shape = image_shape
        
    def _load_data_list(self, data_dir):
        data_list = []

        for f in pathlib.Path(data_dir).glob("*.jpg"):
            data_list.append(f)
        for f in pathlib.Path(data_dir).glob("*.JPG"):
            data_list.append(f)
        for f in pathlib.Path(data_dir).glob("*.png"):
            data_list.append(f)
        for f in pathlib.Path(data_dir).glob("*.PNG"):
            data_list.append(f)
        for f in pathlib.Path(data_dir).glob("*.jpeg"):
            data_list.append(f)
        for f in pathlib.Path(data_dir).glob("*.JPEG"):
            data_list.append(f)

        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = (image.astype(np.float32) - 127.5)/127.5
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image
    
    def __getitem__(self, index):
        path = self.data_list[index]
        image = self._load_image(path)
        return {
            "image": image
        }
