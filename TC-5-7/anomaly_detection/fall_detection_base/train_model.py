import torch, einops, pathlib, cv2, json, shutil
import numpy as np
import matplotlib.pylab as plt
from matplotlib import patches
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from custom_models.custom_models import FallDetector
from custom_models.custom_data import URDataset


EPOCHS = 30
DEVICE = 0
LR = 1e-3


if __name__ == "__main__":
    model = FallDetector()
    model = model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.7, 0.9))
    # train_dataset = URDataset("output/UR_fall_detection3/train")
    # train_dloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # test_dataset = URDataset("output/UR_fall_detection3/test")
    # test_dloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset = URDataset("data/train")
    train_dloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = URDataset("data/test")
    test_dloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for e in range(EPOCHS):
        accs = []
        gts = []
        preds = []
        for frames, labels in train_dloader:
            frames = frames.to(0)
            labels = labels.to(0).float()

            pred = model(frames)
            loss = criterion(pred, labels)

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
        f1_train = f1_score(gts, preds)
        acc_train = sum(accs) / len(accs)

        accs = []
        gts = []
        preds = []

        with torch.inference_mode():
            for frames, labels in test_dloader:
                frames = frames.to(0)
                labels = labels.to(0).float()

                pred = model(frames)
                loss = criterion(pred, labels)

                cat_pred = (pred >= 0.5).float()
                acc = (labels == cat_pred).float().mean()
                accs.append(acc)
                preds.append(cat_pred.detach().cpu().numpy())
                gts.append(labels.cpu().numpy())

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)
        f1_test = f1_score(gts, preds)
        acc_test = sum(accs) / len(accs)

        print(f"(Train) Acc: {acc_train:.4f}, F1: {f1_train:.4f}")
        print(f"(Test) Acc: {acc_test:.4f}, F1: {f1_test:.4f}")

        if f1_test > 0.92 and acc_test > 0.92: break

    torch.save(model.state_dict(), "./fall_detector2.pt")
    # shutil.copy("models/fall_detector2.pt", "falldetection_openpifpaf_custom/fall_detector_with_augmented.pt")

