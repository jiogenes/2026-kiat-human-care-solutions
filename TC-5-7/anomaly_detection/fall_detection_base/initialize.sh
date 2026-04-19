conda create --name fall_detection python=3.7.6
conda activate fall_detection
pip install -r requirements.txt
cp -r falldetection_openpifpaf_custom/* ~/miniconda3/envs/fall_detection/lib/python3.7/site-packages/openpifpaf/

