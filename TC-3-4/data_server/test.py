import time
import requests
import random
import sys
import threading
from multiprocessing import Pool
from functools import partial


URL = "http://localhost:8000/"


def download_video(video, dataset_name):
    try:
        response = requests.get(f"{URL}{dataset_name}/list_frames/train/{video}/")
        # print(f"Status code for the video '{dataset}/{video}': {response.status_code}.")
        frame_list = response.json()
        size = []

        for frame in frame_list:
            response = requests.get(f"{URL}{dataset_name}/get_frame/train/{video}/{frame}", stream=True)
            size.append(len(response.content))

            if response.status_code != 200:
                print(f"ERROR: status code: {response.status_code} for request to '{URL}{dataset_name}/get_frame/train/{video}/{frame}'")

        return sum(size)
    except:
        return 0

def download_sound(data_id, dataset_name):
    try:
        response = requests.get(f"{URL}{dataset_name}/get_{dataset_type}/{data_id}", stream=True)
        return len(response.content)
    except:
        print("ERROR:")
        print(f"\tDATA_NAME: {dataset_name}")
        print(f"\tDATA_TYPE: {dataset_type}")
        print(f"\tDATA_ID: {data_id}")
        return 0


def test(_):
    try:
        response = requests.get(f"{URL}dataset")
        dataset_list = response.json()
        dataset = random.choice(dataset_list)

        dataset_name = dataset["name"]
        dataset_type = dataset["type"]

        if dataset_type == "video":
            response = requests.get(f"{URL}{dataset_name}/list_data/train/")
            print(f"Status code for the dataset '{dataset_name}': {response.status_code}.")

            data_list = response.json()
            data = random.choice(data_list)
            size = download_video(data, dataset_name, dataset_type)
        elif dataset_type == "sound":
            response = requests.get(f"{URL}{dataset_name}/list_data/train/")
            print(f"Status code for the dataset '{dataset_name}': {response.status_code}.")

            data_list = response.json()
            data = random.choice(data_list)
            size = download_sound(data, dataset_name, dataset_type)
        elif dataset_type == "signal":
            response = requests.get(f"{URL}{dataset_name}/get_signal/train/")
            print(f"Status code for the dataset '{dataset_name}': {response.status_code}.")
            size = len(response.content)

        return size
    except:
        return 0


if __name__ == "__main__":
    num_of_proc = 100

    start_time = time.time()

    with Pool(8) as pool:
        sizes = pool.map(test, range(num_of_proc))

    runtime = time.time() - start_time
    print(f"Runtime: {runtime} sec(s).")
    print(f"Total size of processed data: {sum(sizes) / 1_000_000} MB")
    print(f"Throughputs: {sum(sizes) / (runtime * (1_000_000))} MB/s")




# import requests

# # 데이터셋 종류 반환
# requests.get("http://166.104.xxx.xxx:8000/datasets/")

# # 데이터셋 내의 학습용 데이터 종류 반환 (비디오 데이터의 경우)
# requests.get("http://166.104.xxx.xxx:8000/fall_detection_dataset/list_videos/train/")

# # 데이터셋 내의 테스트용 데이터 종류 반환 (비디오 데이터의 경우)
# requests.get("http://166.104.xxx.xxx:8000/fall_detection_dataset/list_videos/test/")

# # 데이터셋 내 학습용 데이터 중, 특정 비디오의 프레임 종류 반환 (비디오 및 이미지 데이터의 경우)
# requests.get("http://166.104.xxx.xxx:8000/fall_detection_dataset/list_frames/train/")

# # 데이터셋 내 학습용 데이터 중, 특정 프레임 반환
# requests.get("http://166.104.xxx.xxx:8000/fall_detection_dataset/get_frame/train/rgb_0425.png")