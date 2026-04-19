rm -rf output/UR_fall_detection
cp -r ./falldetection_openpifpaf_custom/* ~/miniconda3/envs/fall_detection/lib/python3.7/site-packages/openpifpaf/
python run.py --video_dir /data/kiat/UR_fall_detection/parsed --output_dir output/UR_fall_detection