## Training Hf-Depth

If you have already configured the environment and are ready to train the **Hf-Depth** model, please follow the steps below.

---

## 1. Prepare the KITTI Dataset

First, organize the KITTI dataset in the following directory structure:

```id="d2i4tw"
data
├── KITTI
│   ├── splits
│   │   ├── kitti_eigen_train.txt
│   │   ├── kitti_eigen_test.txt
│   │
│   ├── input (RGB, img_dir)
│   │   ├── date_1
│   │   ├── date_2
│   │   ├── ...
│   │
│   ├── gt_depth (ann_dir)
│   │   ├── date_drive_number_sync
│   │   ├── ...
```

### Dataset Description

* **splits/kitti_eigen_train.txt** – training split file
* **splits/kitti_eigen_test.txt** – testing split file
* **input/** – RGB images used as model input
* **gt_depth/** – ground-truth depth maps

Detailed dataset preparation instructions and download links can be found in:

**[Dataset Preparation Guide](../../docs/dataset_prepare.md)**

---

## 2. Select the Training Configuration

To train the model on KITTI, use the following configuration file:

```id="i2o77f"
configs/Hf-Depth/Hf-Depth_swinl_22k_w7_kitti.py
```

---

## 3. Start Training (Command Line)

Run the following command to start training:

```id="xtx2q2"
python tools/train.py configs/Hf-Depth/Hf-Depth_swinl_22k_w7_kitti.py
```

---

## 4. Start Training Without Command-Line Arguments (Optional)

If you prefer not to specify the configuration file each time in the command line, you can modify the default parameter in `tools/train.py`.

Find the following line:

```id="f4c3ux"
parser.add_argument(
    '-config',
    default="configs/Hf-Depth/Hf-Depth_swinl_22k_w7_kitti.py",
    help='train config file path'
)
```

Then simply run:

```id="4xg9sj"
python tools/train.py
```

Training will start automatically.

---

## Training on NYU Dataset

Training on the **NYU dataset** follows the same procedure as KITTI.

Please prepare the dataset according to the instructions in:

**[Dataset Preparation Guide](../../docs/dataset_prepare.md)**

The directory structure and dataset download instructions are provided in that document.

---

## Inference

If you want to run inference with a trained model, please refer to:

**[Inference Guide](../../docs/inference.md)**

This document explains how to run evaluation and visualization using trained checkpoints.

---

## Notes

* Make sure that the dataset paths in the configuration file match your local dataset directories.
* Training checkpoints will be saved in the default **work_dirs/** directory.


