import numpy as np


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