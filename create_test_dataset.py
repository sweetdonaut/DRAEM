import os
import csv
import random
from PIL import Image

PATCH_SIZE = 256
NUM_TRAIN_PATCHES = 10
SEED = 42
CATEGORY = "carpet"


def get_defects_for_image(defects, img_name):
    return [(d['raw_x'], d['raw_y']) for d in defects if d['raw_img_name'] == img_name]


def get_valid_positions(img_size, patch_size, defect_centers, exclusion_size=256):
    positions = []
    step = patch_size

    for y in range(0, img_size - patch_size + 1, step):
        for x in range(0, img_size - patch_size + 1, step):
            patch_cx, patch_cy = x + patch_size // 2, y + patch_size // 2

            valid = True
            for def_x, def_y in defect_centers:
                if (abs(patch_cx - def_x) < (exclusion_size + patch_size) // 2 and
                    abs(patch_cy - def_y) < (exclusion_size + patch_size) // 2):
                    valid = False
                    break

            if valid:
                positions.append((x, y))

    return positions


def crop_patch(img, x, y, patch_size):
    return img.crop((x, y, x + patch_size, y + patch_size))


def crop_center_patch(img, cx, cy, patch_size, img_size=1536):
    half = patch_size // 2
    x = max(0, min(cx - half, img_size - patch_size))
    y = max(0, min(cy - half, img_size - patch_size))
    return img.crop((x, y, x + patch_size, y + patch_size))


def main():
    random.seed(SEED)

    patch_size = PATCH_SIZE
    num_train = NUM_TRAIN_PATCHES

    base_dir = "/home/yclai/vscode_project/DRAEM/datasets"
    src_dir = f"{base_dir}/testing_ebi_raw_img"
    out_dir = f"{base_dir}/testing_ebi_raw_img_{patch_size}/{CATEGORY}"

    os.makedirs(f"{out_dir}/train/good", exist_ok=True)
    os.makedirs(f"{out_dir}/test/good", exist_ok=True)
    os.makedirs(f"{out_dir}/test/bad", exist_ok=True)

    defects = []
    with open(f"{src_dir}/defects.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            defects.append({
                'defect_id': int(row['DEFECTID']),
                'raw_img_name': row['raw_img_name'],
                'raw_x': int(row['raw_x']),
                'raw_y': int(row['raw_y'])
            })

    img_files = sorted([f for f in os.listdir(f"{src_dir}/test/bad") if f.endswith('.png')])

    train_count = 0
    test_good_count = 0
    test_bad_count = 0

    for img_name in img_files:
        img = Image.open(f"{src_dir}/test/bad/{img_name}")
        defect_centers = get_defects_for_image(defects, img_name)

        valid_positions = get_valid_positions(1536, patch_size, defect_centers)

        if len(valid_positions) < num_train + 1:
            print(f"Warning: {img_name} only has {len(valid_positions)} valid positions, need {num_train + 1}")
            actual_train = max(0, len(valid_positions) - 1)
        else:
            actual_train = num_train

        random.shuffle(valid_positions)

        for i in range(actual_train):
            x, y = valid_positions[i]
            patch = crop_patch(img, x, y, patch_size)
            patch.save(f"{out_dir}/train/good/{img_name.replace('.png', '')}_{i:02d}.png")
            train_count += 1

        if len(valid_positions) > actual_train:
            x, y = valid_positions[actual_train]
            patch = crop_patch(img, x, y, patch_size)
            patch.save(f"{out_dir}/test/good/{img_name.replace('.png', '')}_{test_good_count:02d}.png")
            test_good_count += 1

        for def_x, def_y in defect_centers:
            patch = crop_center_patch(img, def_x, def_y, patch_size)
            defect_id = [d['defect_id'] for d in defects
                        if d['raw_img_name'] == img_name and d['raw_x'] == def_x and d['raw_y'] == def_y][0]
            patch.save(f"{out_dir}/test/bad/defect_{defect_id:02d}.png")
            test_bad_count += 1

        print(f"{img_name}: train={actual_train}, defects={len(defect_centers)}")

    print(f"\nDone!")
    print(f"  train/good: {train_count}")
    print(f"  test/good:  {test_good_count}")
    print(f"  test/bad:   {test_bad_count}")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
