# Restoration Network for ProAsPix
This contains the code for the restoration network for ProAsPix.

## Requirements
```
python3
pytorch (tested on version 1.7)
numpy
scipy
matplotlib
```

## Creating datasets
```
cd notebooks
```

Use notebooks `restore_create_data.ipynb` to create training/validation dataset and `restore_create_data_single_image.ipynb` to create test dataset. Make sure to provide correct processed data directory path downloaded from [Box](https://cmu.app.box.com/s/nhhr54dv5is4p65as7rf4uj9shlhbgdg/folder/146406062041).

## Training
```
cd code_train
python restore_train.py
```

## Testing
```
cd notebooks
```
Use `restore_test_save_mat_gpu0.ipynb` to run the saved trained model on test data and save the restored output as a mat file.

## Trained model
Use `restore_0625_data4b_r1_pm1_64x64_Assort_PosEncND_Unet192_1024x1024_NoGuide_rand2022/best.pth` from [Box](https://cmu.app.box.com/s/nhhr54dv5is4p65as7rf4uj9shlhbgdg/folder/146406062041).
