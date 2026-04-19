import torch, einops, pathlib, cv2, json
import numpy as np
from torch.utils.data import Dataset


class URDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = self._load_data_list()

    def _load_data_list(self):
        data_list = []
        for path in pathlib.Path(self.data_dir).glob("keypoints_*.json"):
            data_list.append((path, int("fall" in path.name)))

        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        path, fallen = self.data_list[index]
        frames = self.read_keypoints(path)
        frames = torch.tensor(frames, dtype=torch.float32)
        fallen = torch.tensor(fallen, dtype=torch.long)
        return frames, fallen
    
    def read_keypoints(self, path):
        with open(str(path)) as f:
            data = json.load(f)
            
        keypoints = np.array([kpts[0] for kpts in data["keypoints"] if len(kpts) > 0])
        keypoints += np.random.normal(loc=0, scale=(keypoints.max() - keypoints.min())/100, size=keypoints.shape)
        skeleton = np.array([sks[0] for sks in data["skeleton"] if len(sks) > 0]) - 1
        # print(keypoints.shape, skeleton.shape)
        keypoints = preprocess_keypoints_batch(keypoints, skeleton)
        return keypoints


def preprocess_keypoints(keypoints, skeletons):
    invalid_keypoint_index = keypoints.sum(axis=-1) == 0
    invalid_vector_index = []

    for i in range(len(skeletons)):
        s, e = skeletons[i]
        if s in invalid_keypoint_index or e in invalid_keypoint_index:
            invalid_vector_index.append(i)

    vec = keypoints[skeletons[:, 1]] - keypoints[skeletons[:, 0]]
    vec /= (np.linalg.norm(vec, ord=2, axis=-1, keepdims=True) + 1e-5)
    vec[invalid_vector_index] = 0
    return vec


def preprocess_keypoints_batch(keypoints, skeletons):
    vecs = np.stack([preprocess_keypoints(kpts, sks) for kpts, sks in zip(keypoints, skeletons)])
    return vecs