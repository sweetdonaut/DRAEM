import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import glob
import pandas as pd


def filter_roi_by_border(roi_df, img_shape, scale_x, scale_y, border_margin=0, x_len=10):
    if border_margin <= 0:
        return roi_df

    valid_rows = []
    h, w = img_shape[0], img_shape[1]
    for _, row in roi_df.iterrows():
        cx = int(row['raw_x'] * scale_x)
        cy = int(row['raw_y'] * scale_y)
        half_x = int(x_len * scale_x) // 2
        half_y = int(row['y_len'] * scale_y) // 2
        x1, y1 = cx - half_x, cy - half_y
        x2, y2 = cx + half_x, cy + half_y
        if x1 >= border_margin and y1 >= border_margin and x2 <= w - border_margin and y2 <= h - border_margin:
            valid_rows.append(row)

    if len(valid_rows) == 0:
        return pd.DataFrame(columns=roi_df.columns)
    return pd.DataFrame(valid_rows)


def create_mask(roi_df, img_shape, scale_x, scale_y, dilate_size=0, x_len=10):
    mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
    for _, row in roi_df.iterrows():
        cx = int(row['raw_x'] * scale_x)
        cy = int(row['raw_y'] * scale_y)
        half_x = int(x_len * scale_x) // 2
        half_y = int(row['y_len'] * scale_y) // 2
        x1, y1 = max(0, cx - half_x), max(0, cy - half_y)
        x2, y2 = min(img_shape[1], cx + half_x), min(img_shape[0], cy + half_y)
        mask[y1:y2, x1:x2] = 1.0

    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


class InpaintingTrainDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, dilate_size=0, border_margin=0, x_len=10):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.dilate_size = dilate_size
        self.border_margin = border_margin
        self.x_len = x_len
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "train/good/*.png")))
        self.roi_dir = os.path.join(root_dir, "train_roi/good")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        roi_path = os.path.join(self.roi_dir, f"{filename}.csv")

        image = cv2.imread(img_path)
        roi_df = pd.read_csv(roi_path)

        if self.resize_shape is not None:
            scale_x = self.resize_shape[1] / image.shape[1]
            scale_y = self.resize_shape[0] / image.shape[0]
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        else:
            scale_x, scale_y = 1.0, 1.0

        roi_df = filter_roi_by_border(roi_df, image.shape, scale_x, scale_y, self.border_margin, self.x_len)
        mask = create_mask(roi_df, image.shape, scale_x, scale_y, self.dilate_size, self.x_len)

        image = image.astype(np.float32) / 255.0
        masked_image = image.copy()
        masked_image[mask == 1] = 0

        image = np.transpose(image, (2, 0, 1))
        masked_image = np.transpose(masked_image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return {
            'input': torch.from_numpy(masked_image),
            'target': torch.from_numpy(image),
            'mask': torch.from_numpy(mask)
        }

    def get_dataloader(self, batch_size=8, num_workers=4, shuffle=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


class InpaintingTestDataset(Dataset):

    def __init__(self, root_dir, defect_type, resize_shape=None, dilate_size=0, border_margin=0, x_len=10):
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.dilate_size = dilate_size
        self.border_margin = border_margin
        self.x_len = x_len
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, f"test/{defect_type}/*.png")))
        self.roi_dir = os.path.join(root_dir, f"test_roi/{defect_type}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        roi_path = os.path.join(self.roi_dir, f"{filename}.csv")

        image = cv2.imread(img_path)
        roi_df = pd.read_csv(roi_path)

        if self.resize_shape is not None:
            scale_x = self.resize_shape[1] / image.shape[1]
            scale_y = self.resize_shape[0] / image.shape[0]
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        else:
            scale_x, scale_y = 1.0, 1.0

        roi_df = filter_roi_by_border(roi_df, image.shape, scale_x, scale_y, self.border_margin, self.x_len)
        mask = create_mask(roi_df, image.shape, scale_x, scale_y, self.dilate_size, self.x_len)

        image = image.astype(np.float32) / 255.0
        masked_image = image.copy()
        masked_image[mask == 1] = 0

        image = np.transpose(image, (2, 0, 1))
        masked_image = np.transpose(masked_image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return {
            'input': torch.from_numpy(masked_image),
            'target': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'path': img_path
        }

    def get_dataloader(self, batch_size=8, num_workers=4, shuffle=False):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
