# Cách tải NTU RGB+D dataset

## 1. Đăng ký tải
Vào: https://rose1.ntu.edu.sg/dataset/actionRecognition/
- Điền form, nhận link download qua email
- File cần: NTU_RGBD_skeletons.zip (~32GB cho NTU60)

## 2. Preprocess skeleton data
Sau khi giải nén, dùng script của CTR-GCN để tạo .npy:

```bash
# Clone CTR-GCN để lấy data prep script
git clone https://github.com/Uason-Chen/CTR-GCN
cd CTR-GCN

# Cài dependencies
pip install -r requirements.txt

# Preprocess NTU60 xsub split
python data/ntu60_preprocess.py \
    --skeleton_folder /path/to/nturgbd_skeletons_s001_to_s017/ \
    --out_folder data/ntu60/xsub/ \
    --benchmark xsub \
    --split train

python data/ntu60_preprocess.py \
    --skeleton_folder /path/to/nturgbd_skeletons_s001_to_s017/ \
    --out_folder data/ntu60/xsub/ \
    --benchmark xsub \
    --split val
```

Output files:
- data/ntu60/xsub/train_data.npy  (N, 3, 300, 25, 2)
- data/ntu60/xsub/train_label.pkl
- data/ntu60/xsub/val_data.npy
- data/ntu60/xsub/val_label.pkl

## 3. Chạy PSBA

```bash
cd psba/

# P-PSBA
python train.py --mode p_psba \
    --data_root data/ntu60/xsub \
    --trigger bending_sideways \
    --poison_rate 0.02 \
    --epochs 80 \
    --n_classes 60

# C-PSBA  
python train.py --mode c_psba \
    --data_root data/ntu60/xsub \
    --trigger bending_sideways \
    --poison_rate 0.05 \
    --eps 0.05 \
    --epochs 80 \
    --n_classes 60
```
