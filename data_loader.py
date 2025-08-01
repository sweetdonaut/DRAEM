import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import albumentations as A
from perlin import rand_perlin_2d_np
import tifffile

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, channels=3):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.jpg") + glob.glob(root_dir+"/*/*.png") + glob.glob(root_dir+"/*/*.tiff"))
        self.resize_shape=resize_shape
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        # 檢查是否為 TIFF 檔案
        if image_path.endswith('.tiff'):
            # 使用 tifffile 讀取 32-bit TIFF
            image = tifffile.imread(image_path)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            # 處理通道轉換
            if self.channels == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif self.channels == 1 and image.shape[2] == 3:
                # 灰階轉換保持 float32
                image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=2)
        else:
            # 原有的 cv2.imread 邏輯
            if self.channels == 1:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                # 如果是單通道，增加一個維度
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=2)
            else:  # channels == 3
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                # 如果讀到的是灰階圖，轉換為3通道
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                # 如果已經是彩色圖，保持不變
            
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
            
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], self.channels)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path=None, resize_shape=None, channels=3):
        """
        Args:
            root_dir (string): Directory with all the images.
            anomaly_source_path (string, optional): Path to anomaly source images (e.g., DTD dataset).
                                                   If None, uses random noise.
            resize_shape (list, optional): [height, width] to resize images.
            channels (int): Number of channels (1 or 3).
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.anomaly_source_path = anomaly_source_path
        self.channels = channels

        self.image_paths = sorted(glob.glob(root_dir+"/*.jpg") + glob.glob(root_dir+"/*.png") + glob.glob(root_dir+"/*.tiff"))

        if anomaly_source_path is not None:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg") + glob.glob(anomaly_source_path+"/*/*.tiff"))
            if len(self.anomaly_source_paths) == 0:
                print(f"Warning: No anomaly source images found in {anomaly_source_path}")
                print("Will use random noise instead.")
                self.anomaly_source_paths = None
        else:
            self.anomaly_source_paths = None

        # 根據通道數選擇適合的增強器
        if self.channels == 1:
            # 單通道模式：排除 HueSaturationValue
            self.augmenters = [A.RandomGamma(gamma_limit=(50, 200), p=1.0),
                          A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                          A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                          A.Solarize(p=1.0),
                          A.Posterize(num_bits=4, p=1.0),
                          A.InvertImg(p=1.0),
                          A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                          A.Equalize(p=1.0),
                          A.Rotate(limit=45, p=1.0)
                          ]
        else:
            # 三通道模式：包含所有增強器
            self.augmenters = [A.RandomGamma(gamma_limit=(50, 200), p=1.0),
                          A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                          A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                          A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=0, p=1.0),
                          A.Solarize(p=1.0),
                          A.Posterize(num_bits=4, p=1.0),
                          A.InvertImg(p=1.0),
                          A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                          A.Equalize(p=1.0),
                          A.Rotate(limit=45, p=1.0)
                          ]

        self.rot = A.Rotate(limit=90, p=1.0)


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = A.Compose([self.augmenters[aug_ind[0]],
                         self.augmenters[aug_ind[1]],
                         self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source_path=None):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        if anomaly_source_path is not None:
            # Use external anomaly source (e.g., DTD dataset)
            if anomaly_source_path.endswith('.tiff'):
                # 使用 tifffile 讀取 TIFF
                anomaly_source_img = tifffile.imread(anomaly_source_path)
                if len(anomaly_source_img.shape) == 2:
                    anomaly_source_img = np.expand_dims(anomaly_source_img, axis=2)
                if self.channels == 3 and anomaly_source_img.shape[2] == 1:
                    anomaly_source_img = np.repeat(anomaly_source_img, 3, axis=2)
                elif self.channels == 1 and anomaly_source_img.shape[2] == 3:
                    anomaly_source_img = cv2.cvtColor(anomaly_source_img.astype(np.float32), cv2.COLOR_RGB2GRAY)
                    anomaly_source_img = np.expand_dims(anomaly_source_img, axis=2)
            else:
                if self.channels == 1:
                    anomaly_source_img = cv2.imread(anomaly_source_path, cv2.IMREAD_GRAYSCALE)
                    if len(anomaly_source_img.shape) == 2:
                        anomaly_source_img = np.expand_dims(anomaly_source_img, axis=2)
                else:
                    anomaly_source_img = cv2.imread(anomaly_source_path)
                    # 如果是灰階圖，轉為RGB
                    if len(anomaly_source_img.shape) == 2:
                        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_GRAY2RGB)
            anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            anomaly_img_augmented = aug(image=anomaly_source_img)['image']
        else:
            # Use random noise when no anomaly source is provided
            # Generate random noise based on channel count
            noise = np.random.rand(self.resize_shape[0], self.resize_shape[1], self.channels) * 255
            noise = noise.astype(np.uint8)
            anomaly_img_augmented = aug(image=noise)['image']
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)['image']
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        # 檢查是否為 TIFF 檔案
        if image_path.endswith('.tiff'):
            # 使用 tifffile 讀取 32-bit TIFF
            image = tifffile.imread(image_path)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            # 處理通道轉換
            if self.channels == 3 and image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif self.channels == 1 and image.shape[2] == 3:
                # 灰階轉換保持 float32
                image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
                image = np.expand_dims(image, axis=2)
        else:
            # 原有的 cv2.imread 邏輯
            if self.channels == 1:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=2)
            else:
                image = cv2.imread(image_path)
                # 如果讀到的是灰階圖，轉換為3通道
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)['image']

        image = np.array(image).reshape((image.shape[0], image.shape[1], self.channels)).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        
        if self.anomaly_source_paths is not None:
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            anomaly_source_path = self.anomaly_source_paths[anomaly_source_idx]
        else:
            anomaly_source_path = None
            
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
            self.image_paths[idx], anomaly_source_path)
        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample
