import argparse
from collections import defaultdict
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import prep_splits, get_data_loaders
from model.c3d import C3DInspired
from model.resnet_model import ResNet3D
from test import save_cm, test_model


def train_one_epoch(epoch: int, model: torch.nn.Module, loader: DataLoader,
                    optimizer: torch.optim, criterion: torch.nn.modules.loss,
                    dev: torch.cuda.device):
    running_loss = 0.0
    y_true = []
    y_pred = []
    for i, data in tqdm(enumerate(loader), desc=f"Training epoch-{epoch}", total=len(loader)):
        # data is in format [input, labels, clip-path]
        input, labels, _ = data
        input = input.to(dev)
        labels = labels.to(dev)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # aggergate the loss
        running_loss += loss.item()
        output_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_labels)  # Save Prediction
        y_true.extend(labels.data.cpu().numpy())  # Save Truth
    return y_pred, y_true, running_loss/len(loader)  # average loss per batch


def validate_one_epoch(epoch: int, model: torch.nn.Module, loader: DataLoader,
                       criterion: torch.nn.modules.loss, dev: torch.cuda.device):
    running_loss = 0.0
    y_true = defaultdict(list)
    y_pred = defaultdict(list)
    for i, data in tqdm(enumerate(loader), desc=f"Validating epoch-{epoch}", total=len(loader)):
        # data is in format [input, labels, clip-path]
        input, labels, clip_ = data
        input = input.to(dev)
        labels = labels.to(dev)

        # forward + backward + optimize
        outputs = model(input)
        loss = criterion(outputs, labels)
        # aggergate the loss
        running_loss += loss.item()
        output_labels = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        for pred_, true_, clip__ in zip(output_labels, labels, clip_):
            y_true[clip__].append(true_.data.cpu().numpy())  # Save Truth
            y_pred[clip__].append(pred_)  # Save Prediction
        # print(y_true, y_pred)

    # now find the prediction that happens the most number of times for a clip
    y_pred = [max(y_pred[k], key=y_pred[k].count) for k in y_pred.keys()]
    y_true = [max(y_true[k], key=y_true[k].count) for k in y_true.keys()]
    return y_pred, y_true, running_loss / len(loader)  # average loss per batch


def train(model: torch.nn.Module, root_loc: str, n_epochs: int = 100, board_loc: str = "C3D",
          model_name: str = "C3D.pth", lr: float = None, oversample: bool = False):
    os.makedirs(os.path.join(os.path.dirname(root_loc), "runs"), exist_ok=True)
    writer = SummaryWriter(board_loc)
    train_cms_loc = os.path.join(board_loc, "trainCM")
    os.makedirs(os.path.join(board_loc, "trainCM"), exist_ok=True)
    val_cms_loc = os.path.join(board_loc, "valCM")
    os.makedirs(os.path.join(board_loc, "valCM"), exist_ok=True)

    train_df, val_df, test_df, class_encodings = prep_splits(root_loc=root_loc)
    train_loader, val_loader, test_loader = get_data_loaders(train_df=train_df, val_df=val_df, test_df=test_df,
                                                             encodings=class_encodings,
                                                             with_train_over_sampling=oversample)
    classes = list(class_encodings.values())
    classes.sort()  # encoded - classes are in sorted order

    dummy_input = torch.zeros(1, 3, 30, 224, 224)
    writer.add_graph(model, dummy_input)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")

    model.to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = lr if lr is not None else 0.01
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    best_epoch = -1
    best_val_acc = None
    best_train_acc = None

    for epoch in range(n_epochs):
        model.train(True)
        train_preds, train_labels, train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion, dev)
        train_acc = save_cm(y_pred=train_preds, y_true=train_labels, classes=classes,
                            img_path=os.path.join(train_cms_loc, f"epoch_{epoch+1}.png"))
        # we do not need gradients for validation
        model.train(False)
        val_preds, val_labels, val_loss = validate_one_epoch(epoch, model, val_loader, criterion, dev)
        val_acc = save_cm(y_pred=val_preds, y_true=val_labels, classes=classes,
                          img_path=os.path.join(val_cms_loc, f"epoch_{epoch + 1}.png"))

        if best_val_acc is None:
            best_epoch = epoch
            best_val_acc = val_acc
            best_train_acc = train_acc
        elif val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(board_loc, model_name))
            best_epoch = epoch
            best_val_acc = val_acc
            best_train_acc = train_acc

        print(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
        writer.add_scalars(f"Training vs Validation Loss ({board_loc})",
                           {"train": train_loss, "validation": val_loss},
                           epoch+1)
        writer.add_scalars(f"Training vs Validation Accuracy ({board_loc})",
                           {"train": train_acc, "validation": val_acc},
                           epoch+1)

        # writer.add_image("Training CM", os.path.join(train_cms_loc, f"epoch_{epoch + 1}.png"), epoch+1)
        # writer.add_image("Validation CM", os.path.join(val_cms_loc, f"epoch_{epoch + 1}.png"), epoch + 1)

    print(f"Best Epoch: {best_epoch}, Best Training Accuracy: {best_train_acc*100} %, Best Validation Accuracy: {best_val_acc*100} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", type=str,
                        default="C:\\Users\\transponster\\Documents\\anshul\\task\\dataset-anshul",
                        help="Directory where the data is saved")
    parser.add_argument("--board_loc", "-b", type=str,
                        default="C3DInspired",
                        help="Directory to save model and tensorboard in")
    parser.add_argument("--model_name", "-m", type=str,
                        default="C3D_Inspired.pth", help="name of the model")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="Number of epochs to train for")
    parser.add_argument("--oversample", "-o", type=int, default=0, choices=[0, 1],
                        help="Argument to decide if we want to oversample")
    parser.add_argument("--arch", "-a", type=str, default="C3D", choices=["C3D", "ResNet3D"],
                        help="Model architecture to train")

    args = parser.parse_args()

    root_loc = args.dataset_path
    train_dummy, valid_dummy, test_df, class_encodings = prep_splits(root_loc=root_loc, print_=False)
    n_classes = len(class_encodings.keys())
    board_loc = args.board_loc
    model_name = args.model_name
    oversample = True if args.oversample == 1 else False
    n_epochs = args.epochs
    if args.arch == "C3D":
        model = C3DInspired(n_classes=n_classes)
    elif args.arch == "ResNet3D":
        model = ResNet3D(depth=18, num_classes=n_classes)

    train(model=model, root_loc=root_loc, n_epochs=n_epochs, board_loc=board_loc,
          model_name=model_name, oversample=oversample, lr=1e-2)

    model.load_state_dict(torch.load(os.path.join(board_loc, model_name)))
    _, _, test_loader = get_data_loaders(train_df=train_dummy, val_df=valid_dummy, test_df=test_df,
                                         encodings=class_encodings, with_train_over_sampling=False)
    classes = list(class_encodings.values())
    classes.sort()  # encoded - classes are in sorted order

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    test_acc = test_model(model=model, loader=test_loader,
                          img_path=os.path.join(os.path.dirname(root_loc), "runs", board_loc, "test.png"),
                          classes=classes, dev=dev)

    print(f"Test accuracy of the best model is {test_acc*100} %")
    # model = C3D(n_classes=n_classes)
    # train(model=model, root_loc=root_loc, n_epochs=50, board_loc="C3D_Weighted_Oversample", model_name="C3D_WOS.pth",
    #       lr=1e-4)
