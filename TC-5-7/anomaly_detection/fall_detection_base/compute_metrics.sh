rm -rf output/fall_detection_test
cp -r ./falldetection_openpifpaf_custom/* ~/miniconda3/envs/kiat-2026/lib/python3.11/site-packages/openpifpaf/
python run.py --video_dir data/video/UR_fall_detection/test --output_dir output/fall_detection_test
python run.py --video_dir data/video/fall_detection_augmented_dataset/test --output_dir output/fall_detection_test
python compute_metrics.py --output_dir output/fall_detection_test