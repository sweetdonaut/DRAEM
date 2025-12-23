import os
import numpy as np
import cv2
import pandas as pd
import glob
from tqdm import tqdm


def get_rois_in_crop(roi_df, crop_x, crop_y, crop_size, x_len=10):
    valid_rows = []
    for _, row in roi_df.iterrows():
        cx, cy = row['raw_x'], row['raw_y']
        half_x = x_len // 2
        half_y = row['y_len'] // 2
        x1, y1 = cx - half_x, cy - half_y
        x2, y2 = cx + half_x, cy + half_y

        if x1 >= crop_x and y1 >= crop_y and x2 <= crop_x + crop_size and y2 <= crop_y + crop_size:
            new_row = row.copy()
            new_row['raw_x'] = cx - crop_x
            new_row['raw_y'] = cy - crop_y
            valid_rows.append(new_row)

    if len(valid_rows) == 0:
        return None
    return pd.DataFrame(valid_rows)


def crop_dataset(args):
    image_paths = sorted(glob.glob(os.path.join(args.input_path, "train/good/*.png")))
    roi_dir = os.path.join(args.input_path, "train_roi/good")

    out_img_dir = os.path.join(args.output_path, "train/good")
    out_roi_dir = os.path.join(args.output_path, "train_roi/good")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_roi_dir, exist_ok=True)

    crop_count = 0
    for img_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        roi_path = os.path.join(roi_dir, f"{filename}.csv")

        image = cv2.imread(img_path)
        roi_df = pd.read_csv(roi_path)

        h, w = image.shape[:2]
        max_x = w - args.crop_size
        max_y = h - args.crop_size

        if max_x < 0 or max_y < 0:
            print(f"Skip {filename}: image smaller than crop size")
            continue

        for _ in range(args.num_crops):
            crop_x = np.random.randint(0, max_x + 1)
            crop_y = np.random.randint(0, max_y + 1)

            new_roi_df = get_rois_in_crop(roi_df, crop_x, crop_y, args.crop_size, args.x_len)

            if new_roi_df is None or len(new_roi_df) < args.min_rois:
                continue

            cropped_img = image[crop_y:crop_y + args.crop_size, crop_x:crop_x + args.crop_size]

            out_name = f"{filename}_{crop_count:04d}"
            cv2.imwrite(os.path.join(out_img_dir, f"{out_name}.png"), cropped_img)
            new_roi_df.to_csv(os.path.join(out_roi_dir, f"{out_name}.csv"), index=False)
            crop_count += 1

    print(f"Generated {crop_count} cropped images")


if __name__ == "__main__":
    class Args:
        input_path = "datasets/carpet_gray"
        output_path = "datasets/carpet_gray_256"
        crop_size = 256
        num_crops = 10
        min_rois = 1
        x_len = 10

    crop_dataset(Args())
