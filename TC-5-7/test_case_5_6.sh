cd /home/jyji/develop/2024-kiat-wonderful/TC-5-7/anomaly_detection/fall_detection_base
rm -rf output/fall_detection_test
cp -r ./falldetection_openpifpaf_custom/* ~/miniconda3/envs/kiat/lib/python3.11/site-packages/openpifpaf/
python run.py --video_dir data/video/UR_fall_detection/test --output_dir output/fall_detection_test
python run.py --video_dir data/video/fall_detection_augmented_dataset/test --output_dir output/fall_detection_test
python compute_metrics.py --output_dir output/fall_detection_test
