import torch, einops, pathlib, cv2, json, shutil
import numpy as np
from typing import List, Tuple, Literal, Optional

import matplotlib.pylab as plt
from matplotlib import patches
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from custom_models.custom_models import FallDetector
from custom_models.custom_data import URDataset


EPOCHS = 30
DEVICE = 0
LR = 1e-5

def train(train_loader, model, test_loader=None, epochs=EPOCHS, lr=LR, device=DEVICE, train_only=False, save=True, save_path="models/fall_detector.pt", log=True):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.9)) # betas=(0.5, 0.9)
    for e in range(epochs):
        losses = []
        accs = []
        f1s = []
        gts = []
        preds = []
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device).float()

            pred = model(frames)
            loss = criterion(pred, labels)

            losses.append(loss.item())
            cat_pred = (pred >= 0.5).float()
            acc = (labels == cat_pred).float().mean()
            accs.append(acc)
            preds.append(cat_pred.detach().cpu().numpy())
            gts.append(labels.cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)
        loss = sum(losses) / len(losses)
        f1 = f1_score(gts, preds)
        acc = sum(accs) / len(accs)
        if log:
            print(f"(Train) Acc: {acc:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")
        if f1 > 0.80 and acc > 0.80:
            break # 오버피팅 방지

        if not train_only:
            eval_return = evaluate(test_loader, model, criterion)
            # if eval_return['f1'] > 0.9 and eval_return['acc'] > 0.9:
            #     break

    if save:
        torch.save(model.state_dict(), save_path)
        shutil.copy(save_path, "falldetection_openpifpaf_custom/fall_detector.pt")

    return {
        'loss': loss,
        'acc': acc,
        'f1': f1,
    }

@torch.no_grad()
def evaluate(loader, model, criterion=None, device=DEVICE, log=True):
    if criterion is None:
        criterion = nn.BCELoss()
    accs = []
    gts = []
    preds = []
    losses = []
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device).float()
        pred = model(frames)
        loss = criterion(pred, labels)
        losses.append(loss.item())

        cat_pred = (pred >= 0.5).float()
        acc = (labels == cat_pred).float().mean()
        accs.append(acc)
        preds.append(cat_pred.detach().cpu().numpy())
        gts.append(labels.cpu().numpy())

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    f1 = f1_score(gts, preds)
    acc = sum(accs) / len(accs)
    loss = sum(losses) / len(losses)

    if log:
        print(f"(Test) Acc: {acc:.4f}, F1: {f1:.4f}, Loss: {loss:.4f}")

    return {
        'acc': acc.item(),
        'f1': f1,
        'loss': loss,
    }

def binary_entropy_nats(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: sigmoid 확률 (0~1), 엔트로피 단위: nats (기본)
    p = p.clamp(eps, 1 - eps)
    entropy = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
    return entropy

def uncertainty_score(prob: torch.Tensor,
                      metric: Literal["entropy", "margin", "hybrid"] = "hybrid") -> torch.Tensor:
    """
    prob: shape [B, ...]의 확률(시그모이드 출력)
    - entropy: 최대 ~ ln(2) ≈ 0.693 nats
    - margin: u = 0.5 - |p - 0.5|  (0~0.5), 값이 클수록 불확실
    """

    if metric == "entropy": # 0~1
        u = binary_entropy_nats(prob) / np.log(2)
    elif metric == "margin": # 0~1
        u = 0.5 - (prob - 0.5).abs()
    elif metric == "hybrid": # 0~1
        ent = binary_entropy_nats(prob) / np.log(2)
        mar = (0.5 - (prob - 0.5).abs()) / 0.5
        u = 0.5 * ent + 0.5 * mar
    return u.mean()

@torch.no_grad()
def get_lowest_uncertainty_index(model: nn.Module, dataset: Dataset) -> int:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    min_uncertainty = float('inf')
    min_index = -1
    for i, (frames, labels) in enumerate(loader):
        frames = frames.to(DEVICE)
        probs = model(frames)
        uncertainty = uncertainty_score(probs).item()
        if uncertainty < min_uncertainty:
            min_uncertainty = uncertainty
            min_index = i
    return min_index


def active_learning(model: nn.Module,
                    train_dataset: Dataset,
                    al_dataset: Dataset,
                    test_loader: DataLoader,
                    device=DEVICE,
                    threshold: float = 0.50,
                    metric: Literal["entropy", "margin"] = "entropy",
                    lr: float = LR):

    added_count = 0
    al_loader = DataLoader(al_dataset, batch_size=1, shuffle=False)
    for _ in range(EPOCHS):
        for i, (frames, labels) in enumerate(al_loader):
            frames = frames.to(device)
            labels = labels.to(device).float() # 원래는 없어야됨

            probs = model(frames)  # sigmoid 확률
            uncertainty = uncertainty_score(probs, metric=metric).item()
            # print(uncertainty)

            # base + memory로 1 epoch 미세학습
            if uncertainty >= threshold:
                # 라벨을 받았다고 가정 후 al_dataset의 정답 레이블 사용
                added_count += 1
                print(f"[AL] Added sample #{added_count} (uncertainty={uncertainty:.4f} {metric})")

                # 10% 확률로 가장 불확실한 샘플 제거
                if np.random.random() < 0.1:
                    lowest_index = get_lowest_uncertainty_index(model, train_dataset)
                    train_dataset = Subset(train_dataset, [i for i in range(len(train_dataset)) if i != lowest_index])

                # al dataset 에서 현재 샘플 추가
                al_subset = Subset(al_dataset, [i])

                train_dataset = ConcatDataset([train_dataset, al_subset])
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
                ft_metrics = train(train_loader, model, epochs=1, lr=lr, device=device, train_only=True, save=False, log=False)
                eval_metrics = evaluate(test_loader, model, device=device, log=False)
                print(f"| Train(acc={ft_metrics['acc']:.4f}, f1={ft_metrics['f1']:.4f}, loss={ft_metrics['loss']:.4f})\n"
                    f"| Test(acc={eval_metrics['acc']:.4f}, f1={eval_metrics['f1']:.4f}, loss={eval_metrics['loss']:.4f})")

                if eval_metrics['f1'] >= 0.70 and eval_metrics['acc'] >= 0.90:
                    print(f"[AL] Reached target F1 score. added_count:{added_count}, active rate: {added_count/len(al_dataset):.2f}")

                if added_count / len(al_dataset) >= 0.7:
                    print(f"[AL] Reached target active rate. active rate: {added_count/len(al_dataset):.2f}")
                    break

    # 최종 저장
    # pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), "models/fall_detector_al.pt")
    # shutil.copy("models/fall_detector_al.pt", "falldetection_openpifpaf_custom/fall_detector.pt")
    print(f"[AL] Done. Added {added_count} samples. Active rate: {added_count/len(al_dataset):.2f} Accuracy : {eval_metrics['acc']}  F1 score: {eval_metrics['f1']}")
    return model


if __name__ == "__main__":
    train_dataset = URDataset("output/fall_detection_al_dataset/train")
    al_dataset = URDataset("output/fall_detection_al_dataset/valid")
    test_dataset = URDataset("output/fall_detection_al_dataset/test")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    al_loader = DataLoader(al_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = FallDetector().to(DEVICE)
    model.load_state_dict(torch.load("fall_detector2.pt"))

    active_learning(
        model,
        train_dataset=train_dataset,
        al_dataset=al_dataset,
        test_loader=test_loader,
        device=DEVICE,
        threshold=0.2,
        lr=LR
    )