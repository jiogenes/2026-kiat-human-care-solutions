import cv2, pathlib


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def make_video(path):
    frame_path_list = []
    frame_path_list.extend(list(pathlib.Path(path).glob("*.jpg")))
    frame_path_list.extend(list(pathlib.Path(path).glob("*.JPG")))
    frame_path_list.extend(list(pathlib.Path(path).glob("*.png")))
    frame_path_list.extend(list(pathlib.Path(path).glob("*.PNG")))
    frame_path_list.extend(list(pathlib.Path(path).glob("*.jpeg")))
    frame_path_list.extend(list(pathlib.Path(path).glob("*.JPEG")))
    frame_path_list.sort(key=lambda p: p.name)

    frame_list = list(map(lambda p: read_image(str(p))))

