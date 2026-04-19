import subprocess, argparse, os, pathlib
import xml.etree.ElementTree as ET
from tqdm import tqdm


def copy_config(default_config_path, target_config_path, video_path):
    tree = ET.parse(default_config_path)
    root = tree.getroot()
    for child in root.iter("RTSPURL"):
        child.text = video_path

    tree.write(target_config_path)


def run(video_input, video_output, openpifpaf_dir, default_config_path):
    target_config_path = f"{openpifpaf_dir}/config/config.xml"

    default_config_path = os.path.abspath(default_config_path)
    target_config_path = os.path.abspath(target_config_path)
    video_input = os.path.abspath(video_input)

    copy_config(default_config_path,
                target_config_path,
                video_input)

    subprocess.run([
        "python",
        "-m",
        "openpifpaf.video",
        "--video-output",
        f"{video_output}"
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_input", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--openpifpaf_dir", type=str, default=f"{os.environ['CONDA_PREFIX']}/lib/python3.11/site-packages/openpifpaf")
    parser.add_argument("--default_config_path", type=str, default="falldetection_openpifpaf/config/config.xml")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.video_dir is not None or args.video_input is not None
    os.makedirs(args.output_dir, exist_ok=True)

    if args.video_input is None:
        video_paths = list(pathlib.Path(args.video_dir).glob("*.mp4"))
        for video_input_path in tqdm(video_paths, desc="Processing videos", unit="video"):
            video_name = video_input_path.name
            video_output = os.path.join(args.output_dir, video_name)
            run(str(video_input_path), video_output, args.openpifpaf_dir, args.default_config_path)
    else:
        video_name = args.video_input.split("/")[-1]
        video_output = os.path.join(args.output_dir, video_name)
        run(args.video_input, video_output, args.openpifpaf_dir, args.default_config_path)
