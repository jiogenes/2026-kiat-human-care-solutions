import cv2, glob, argparse


def run(args):
    # Get all PNG files in the folder, sorted by name
    images = sorted(glob.glob(f"{args.image_dir}/*.png"))

    # Read the first image to get the frame size
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.video_path, fourcc, args.fps, (width, height))

    for image in images:
        frame = cv2.imread(image)
        video.write(frame)

    video.release()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())