from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_cm(y_pred: list, y_true: list, classes: list, img_path: str = "") -> float:
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(img_path)
    plt.close()
    return accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)


def test_model(model: torch.nn.Module, loader: DataLoader, img_path: str, classes: list, dev: torch.device) -> float:
    y_true = defaultdict(list)
    y_pred = defaultdict(list)

    model.to(dev)
    model.train(False)
    # iterate over test data
    for i, data in tqdm(enumerate(loader), desc=f"Testing Model Performance", total=len(loader)):
        # data is in format [input, labels, clip-path]
        input, labels, clip_ = data
        input = input.to(dev)
        labels = labels.to(dev)
        output = model(input)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        for pred_, true_, clip__ in zip(output, labels, clip_):
            y_true[clip__].append(true_)  # Save Truth
            y_pred[clip__].append(pred_)  # Save Prediction

    # now find the prediction that happens the most number of times for a clip
    y_pred = [max(y_pred[k], key=y_pred[k].count) for k in y_pred.keys()]
    y_true = [max(y_true[k], key=y_true[k].count) for k in y_true.keys()]

    # save the CM for the test-loader
    acc_score = save_cm(y_pred=y_pred, y_true=y_true, classes=classes, img_path=img_path)
    return acc_score
