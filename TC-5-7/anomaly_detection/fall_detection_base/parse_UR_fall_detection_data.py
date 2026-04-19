import cv2, pathlib, argparse
import numpy as np
import matplotlib.pylab as plt


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames), fps

def parse_frames(frames):
    height, width, layers = frames[0].shape
    frames = frames[:, :, width//2:]
    return frames

def write_video(frames, fps, path):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="directory path including mp4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="video output path to store parsed video")
    args = parser.parse_args()

    for path in pathlib.Path(args.input_dir).glob("*.mp4"):
        frames, fps = read_video(str(path))
        frames = parse_frames(frames)
        write_video(frames, fps, f"{args.output_dir}/parsed_{path.name}")