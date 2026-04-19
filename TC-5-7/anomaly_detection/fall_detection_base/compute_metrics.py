import pathlib, argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(output_dir):
    preds = []
    gts = []

    for txt_file in pathlib.Path(output_dir).glob("prediction_final_*.txt"):
        with open(str(txt_file)) as f:
            lines = f.readlines()

        preds.append(int(lines[0]))

        if "fall" in txt_file.name:
            gts.append(1)
        else:
            gts.append(0)

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds)
    print(f"Acc: {acc:.8f}, F1: {f1:.8f}")


if __name__ == "__main__":
    """
    UR fall detection dataset (https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
    Acc: 0.93023256
    F1: 0.92105263
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/UR_fall_detection4")
    args = parser.parse_args()

    compute_metrics(args.output_dir)
