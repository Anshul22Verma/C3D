import collections

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from torchvideotransforms import video_transforms, volume_transforms


def prep_splits(root_loc: str, print_: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if not os.path.exists(os.path.join(root_loc, "log_classes.csv")):
        classes_ = [loc for loc in os.listdir(root_loc) if os.path.isdir(os.path.join(root_loc, loc))]

        df = pd.DataFrame()
        files = []
        l_classes = []

        for cls in classes_:
            for vid in os.listdir(os.path.join(root_loc, cls)):
                files.append(os.path.join(root_loc, cls, vid))
                l_classes.append(cls)
        df["files"] = files
        df["class"] = l_classes

        # fixing the random stater to get the same split everytime
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=2023, stratify=l_classes)
        # add some validation samples to track training over-fitting
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2023,
                                            stratify=train_df["class"].values.tolist())

        if len(val_df[val_df["class"] == "00"]) == 0:
            # just to have at-least one sample from all classes in all the sets
            # moving a record of class "00" in the validation set for this class if it's not present
            aux_df_list = train_df.index[train_df["class"] == "00"].tolist()
            row = train_df.loc[aux_df_list[0]]
            train_df = train_df.drop([aux_df_list[0]])
            val_df = val_df.append(row.to_dict(), ignore_index=True)

        train_df["split"] = ["train" for _ in range(len(train_df))]
        val_df["split"] = ["val" for _ in range(len(val_df))]
        test_df["split"] = ["test" for _ in range(len(test_df))]

        df = pd.concat([train_df, val_df, test_df])
        df.to_csv(os.path.join(root_loc, "log_classes.csv"), index=False)
    else:
        # the split already exists we can just load it
        df = pd.read_csv(os.path.join(root_loc, "log_classes.csv"))
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]

    if print_:
        print(f"Unique class distribution in train set: {np.unique(train_df['class'], return_counts=True)}")
        print(f"Unique class distribution in validation set: {np.unique(val_df['class'], return_counts=True)}")
        print(f"Unique class distribution in test set: {np.unique(test_df['class'], return_counts=True)}")

    classes = list(np.unique(df["class"].values.tolist()))
    classes.sort()
    class_encodings = {}
    for idx, cls in enumerate(classes):
        class_encodings[cls] = idx

    return train_df, val_df, test_df, class_encodings


def load_sample(f_path: str, sampling: int = 0) -> list:
    """
    :param f_path: a video with .avi extension
    :param sampling: to sample frames options [0, 1, 3, 4, 5, 6, 7, 8, 9]
                     ** To make the test prediction all the 9-sampling result is combined **
    :return: a numpy array of shape frames X H X W X C
    """
    cap = cv2.VideoCapture(f_path)

    frames = []
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            last_frame = Image.fromarray(frame)
            frames.append(Image.fromarray(frame))
        else:
            break
    # there are some videos with 294 frames some others might have more so just adding this for caution
    if len(frames) > 300:
        frames = frames[:300]
    while len(frames) != 300:
        frames.append(last_frame)
    cap.release()
    frames = [frame for i, frame in enumerate(frames) if i % 10 == sampling]
    return frames


def get_transforms(train: bool = True, mean: tuple = None, std: tuple = None) -> video_transforms.Compose:
    if train:
        transforms__ = [
            video_transforms.Resize((224, 224)),
            video_transforms.RandomGrayscale(p=0.4),
            video_transforms.RandomHorizontalFlip(p=0.3),
            video_transforms.RandomRotation(degrees=15),
            video_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # volume_transforms.ClipToTensor()  # this scales the image as well /255
        ]
        if mean is None or std is None:
            transforms__.append(
                volume_transforms.ClipToTensor()
            )
        else:
            transforms__.append(
                video_transforms.Normalize(mean=mean, std=std),
                volume_transforms.ClipToTensor(div_255=False)
            )
    else:
        transforms__ = [
            video_transforms.Resize((224, 224)),
        ]
        if mean is None or std is None:
            transforms__.append(
                volume_transforms.ClipToTensor()
            )
        else:
            transforms__.append(
                video_transforms.Normalize(mean=mean, std=std),
                volume_transforms.ClipToTensor(div_255=False)
            )
    return video_transforms.Compose(transforms__)


def save_transformed_video(vid: torch.Tensor, f_path: str, fps: int = 10, size: tuple = (224, 224)) -> None:
    writer = cv2.VideoWriter(f_path, 0, fps, size)
    vid = vid.cpu().detach().numpy()
    vid_ = vid.transpose((1, 2, 3, 0))
    for frame in vid_:
        # import matplotlib.pyplot as plt
        # plt.imshow(frame)
        # plt.show(block=True)
        writer.write((frame * 255).astype(np.uint8))
    writer.release()


class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, encodings: dict, sample: int = 0):
        super(VideoDataset, self).__init__()
        self.df = df
        self.transforms = get_transforms(train=train)
        self.encodings = encodings
        self.is_train = train
        self.sample = sample

    def __getitem__(self, index):
        row_ = self.df.iloc[index]
        clip_path = row_["files"]
        clip_label = row_["class"]

        # sample a random frame of every 10 frames
        if self.is_train:
            self.sample = np.random.randint(0, 9)

        clip_ = load_sample(f_path=clip_path, sampling=self.sample)  # get the video with 30 frames

        # transform clips
        tensor_clip = self.transforms(clip_)

        return tensor_clip, torch.tensor(self.encodings[clip_label]), clip_path

    def __len__(self):
        return len(self.df)


def get_data_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, encodings: dict,
                     with_train_over_sampling: bool = False):
    train_dataset = VideoDataset(df=train_df, train=True, encodings=encodings)
    val_dataset = ConcatDataset([
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=0),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=1),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=2),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=3),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=4),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=5),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=6),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=7),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=8),
        VideoDataset(df=val_df, train=False, encodings=encodings, sample=9),
    ])
    test_dataset = ConcatDataset([
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=0),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=1),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=2),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=3),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=4),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=5),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=6),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=7),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=8),
        VideoDataset(df=test_df, train=False, encodings=encodings, sample=9),
    ])

    if not with_train_over_sampling:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        cls_, counts_ = np.unique(train_df["class"].values.tolist(), return_counts=True)
        # need to assign higher weight to smaller class
        counts_ = [c if c != 2 else 10 for c in counts_]
        class_weights = {k: sum(counts_)/c for k, c in zip(cls_, counts_)}

        # weight of class 0 becomes vey dominant because it has the very few classes
        # therefore I want to reduce the oversampling, still oversampling but not to an extent to balance everything
        print(f"Class weights for oversampling: {class_weights}")
        weight_for_each_sample = [class_weights[c] for c in train_df["class"].values.tolist()]

        sampler = WeightedRandomSampler(weight_for_each_sample, num_samples=int(max(counts_)*len(cls_)))  # len(train_df))
        train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_df, val_df, test_df, class_encodings = prep_splits(
        root_loc="C:\\Users\\transponster\\Documents\\anshul\\task\\dataset-anshul")

    # frames_0 = load_sample(f_path="C:\\Users\\transponster\\Documents\\anshul\\task\\dataset-anshul\\01\\9877360270.avi",
    #                       sampling=0)
    # frames_9 = load_sample(f_path="C:\\Users\\transponster\\Documents\\anshul\\task\\dataset-anshul\\01\\9877360270.avi",
    #                       sampling=9)
    # transforms_ = get_transforms(train=False)
    #
    # clip = transforms_(frames_0)
    # save_transformed_video(vid=clip, f_path="C:\\Users\\transponster\\Desktop\\test_0.avi")
    # clip = transforms_(frames_9)
    # save_transformed_video(vid=clip, f_path="C:\\Users\\transponster\\Desktop\\test_9.avi")
    train_loader, val_loader, test_loader = get_data_loaders(train_df=train_df, val_df=val_df, test_df=test_df,
                                                             encodings=class_encodings, with_train_over_sampling=True)
    clss_count = collections.defaultdict(int)
    for data_ in train_loader:
        _, ls, _ = data_
        for l in ls.data.cpu().numpy():
            clss_count[l] += 1
        print(clss_count)
    print(clss_count)
