# 낙상감지 모듈

### 모듈 인터페이스
- 입력 (다음 중 하나)
    - 비디오(.mp4)파일들이 담긴 디렉토리
    - 단일 비디오(.mp4) 파일
- 출력
    - 시각화용 비디오(.mp4) 파일
    - 낙상예측 txt 파일 (비디오마다 생성됨)
        - 빈 파일이 아니면 **"낙상일어남"**을 의미 (파일 내 적힌 숫자는 낙상 일어난 프레임 번호임)
        - 빈 파일이면 **"낙상 일어나지 않음"**을 의미

### 실행환경 설치
다음을 차례대로 실행한다. 이때, `{ENV_NAME}`는 원하는 이름 넣으면 되고(예: `fall_detection`), 다음 명령어에서 `{ENV_NAME}`를 모두 지우고 그 자리에 해당 이름을 넣어준다.

```sh
conda create -n {ENV_NAME} python=3.7.6
conda activate {ENV_NAME}
pip install -r requirements.txt
cp -r falldetection_openpifpaf_custom/* ~/miniconda3/envs/{ENV_NAME}/lib/python3.7/site-packages/openpifpaf/
```

### 소프트웨어 요구사항
Ubuntu 20.04 이상 운영체제 또는 동일 시기(약 2020년) 이후에 나온 리눅스를 사용한다고 가정한다. Python 3.7.6을 기반으로 동작하고, 필요한 파이썬 패키지는 `requirements.txt` 안에 명시되어 있다.
- python 3.7.6
- torch 1.13.1
- torchvision 0.14.1
- openpifpaf 0.11.9
- matplotlib 3.1.3
- opencv-python 4.2.0.34
- xmltodict 0.12.0
- einops 0.6.1
만약, 위 "실행환경"을 따라서 설치했다면, 별도로 설치할 필요는 없다.

### 데이터준비
다음 과정을 거쳐서 실행한다. (`fall_detection_base` 폴더 내에서 실행)
1. 다음 스크립트로 UR_fall_detection_data를 다운로드 한다. (`download_UR_fall_detection_dataset.sh` 파일 5, 11번째줄에서 다운로드 경로 수정필요. `/data/kiat/UR_fall_detection` 부분 수정하면 됨)
    ```sh
    sh download_UR_fall_detection_dataset.sh
    ```
2. UR_fall_detection_data 비디오는 RGB scene과 depth scene이 width 방향으로 concatenate되어 있기에, 다음 스크립트를 통해 RGB만 추출.
    ```sh
    python parse_UR_fall_detection_data.py --input_dir {UR_FALL_DETECTION_DATA_DIR} --output_dir {PARSED_DIR}
    ```
    `{UR_FALL_DETECTION_DATA_DIR}`는 1번에서 수정한 경로(mp4가 직접 담긴 디렉토리 경로) 기입


### Active learning
- **인터페이스**: 학습 대상 모델은 `fall_detection_base/custom_models/custom_models.py`에 있는 `FallDetector`를 사용하며 다음의 인터페이스를 가짐
    - 입력: 비디오 1개에서 추출한 keypoint
    - 출력: 0~1 사이 확률값 (0.5 이상이면 낙상, 아니면 정상, 비디오 당 1개 float 값 출력)
- **Active learning**: float 1개가 아웃풋인 binary classification이므로, binary entropy를 쓰던지 하면 될듯
- **학습코드**: `fall_detection_base/train_model.py` 참고 (프로젝트 루트 디렉토리)

**입력 데이터 관련:**
- 데이터 다운로드는 아래 "데이터 준비" 참고
- **키포인트 획득**: 입력 키포인트는 다음 코드로 실행하면 output_dir에 `keypoints_parsed_*.json` 이름으로 생성됨
    ```sh
    python run.py --video_dir {PARSED_DIR} --output_dir output/UR_fall_detection
    ```
- **키포인트 전처리**: `keypoints_parsed_*.json` 파일들이 담겨있는 폴더를 `fall_detection_base/custom_models/custom_data.py`에 있는 `URDataset`의 생성자로 넘겨주면 모델이 입력받는 형태로 반환해줌
  - 세부 전처리 코드는 `fall_detection_base/custom_models/custom_data.py`에 있는 `preprocess_keypoints` 함수 참고
- **체크포인트**: `fall_detection_base`폴더에 있는 `fall_detector.pt`. 업데이트했으면 해당 파일을 falldetection_openpifpaf_custom에도 복사해줘야 함 (둘 다 있어야 해서 옮기지 말고 복사)


주의: `fall_detection_base/falldetection_openpifpaf` 는 건들지 말것

### Metric 계산
1. Active learning 수행하고 모델 튜닝. 체크포인트는 `fall_detection_base/fall_detector.pt`와 `fall_detection_base/falldetection_openpifpaf_custom/fall_detector.pt` 에 저장
2. `cd fall_detection_base`
3. `cp -r ./falldetection_openpifpaf_custom/* ~/miniconda3/envs/{ENV_NAME}/lib/python3.7/site-packages/openpifpaf/` 실행
    - python version 다르게 했으면 python3.7도 수정해야 함
4. falldetection 코드 실행
    ```sh
    python run.py --video_dir {PARSED_DIR} --output_dir output/UR_fall_detection
    ```
5. `fall_detection_base/compute_metrics.py` 으로 메트릭 계산
    ```sh
    python compute_metrics.py --video_dir output/UR_detection_dataset
    ```
ㄷ
### 실행 (UR_fall_detection_data인 경우)
1. 다음 코드로 실행
    ```sh
    python run.py --video_dir {PARSED_DIR} --output_dir output/UR_fall_detection
    ```

### 실행 (UR_fall_detection_data 외 다른 비디오인 경우)
1. 스크립트의 입력으로 다음중 하나를 준비 (둘 중 하나만)
    - **비디오(.mp4)파일들이 담긴 디렉토리**
    - **단일 비디오(.mp4) 파일** 
2. 다음 코드로 실행 (`fall_detection_base` 폴더 내에서 실행)
    ```sh
    # 비디오파일들이 담긴 디렉토리일 경우
    python run.py --video_dir {VIDEO_DIR_PATH} --output_dir output/UR_fall_detection
    # 단일 비디오 파일일 경우
    python run.py --video_input {VIDEO_PATH} --output_dir output/UR_fall_detection
    ```

### 학습
-  `train_model.py` 참고. 모델은 `custom_models/custom_models.py`에 있는 `FallDetector`를 사용하며 다음의 인터페이스를 가짐
    - 입력: 비디오 1개에서 추출한 keypoint
    - 출력: 0~1 사이 확률값 (0.5 이상이면 낙상, 아니면 정상, 비디오 당 1개 float 값 출력)
- 키포인트 획득: 입력 키포인트는 위 실행 방법대로 실행하면 output directory에 `keypoints_parsed_*.json` 이름으로 생성됨
- 키포인트 전처리: `keypoints_parsed_*.json` 파일들이 담겨있는 폴더를 `custom_models/custom_data.py`에 있는 `URDataset`의 생성자로 넘겨주면 모델이 입력받는 형태로 반환해줌
  - 세부 전처리 코드는 `custom_models/custom_data.py`에 있는 `preprocess_keypoints` 함수 참고
- 체크포인트는 프로젝트 루트폴더에 있는 `fall_detector.pt`. 업데이트했으면 해당 파일을 falldetection_openpifpaf_custom에도 복사해줘야 함 (둘 다 있어야 해서 옮기지 말고 복사)