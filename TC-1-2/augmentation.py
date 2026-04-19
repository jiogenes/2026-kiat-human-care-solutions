import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def run_module(folder, metrics):
    result = subprocess.run(
        [sys.executable, 'run.py', '--metrics'] + metrics + ['--json'],
        stdout=subprocess.PIPE, stderr=None,  # stderr 상속 → tqdm 터미널에 표시
        text=True,
        cwd=os.path.join(BASE, folder)
    )
    if result.returncode != 0:
        print(f"[{folder}] 오류 발생", file=sys.stderr)
        return {}
    # stdout에 JSON 외 다른 출력이 섞일 수 있으므로 마지막 줄만 파싱
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', nargs='+', choices=['accuracy', 'f1'],
                        default=['accuracy', 'f1'],
                        help='출력할 지표 선택 (기본값: accuracy f1)')
    args = parser.parse_args()

    print("Evaluating respiratory_aug ...")
    resp = run_module('respiratory_aug', args.metrics)

    print("Evaluating pressure_aug ...")
    press = run_module('pressure_aug', args.metrics)

    print()
    if 'f1' in args.metrics:
        print(f"Respiratory F1 Score : {resp.get('f1', float('nan')):.4f}")
        print(f"Pressure    F1 Score : {press.get('f1', float('nan')):.4f}")
    if 'accuracy' in args.metrics:
        print(f"Respiratory Accuracy : {resp.get('accuracy', float('nan')):.4f}")
        print(f"Pressure    Accuracy : {press.get('accuracy', float('nan')):.4f}")


if __name__ == '__main__':
    main()
