# STLGRU: Spatio-Temporal Lightweight Graph GRU for Traffic Flow Prediction

![STLGRU](figure/archnew.png "Model Architecture")

This is the official implementation of STLGRU: Spatio-Temporal Lightweight Graph GRU for Traffic Flow Prediction: \
Kishor Kumar Bhaumik, Fahim Faisal Niloy, Saif mahmud and Simon S. Woo [STLGRU: Spatio-Temporal Lightweight Graph GRU for Traffic Flow Prediction](https://arxiv.org/pdf/2212.04548.pdf).

Dependency can be installed using the following command:
```bash
pip install -r requirement.txt
```


## Data Preparation
Download the dataset(PEMS03, PEMS04, PEMS07, PEMS08) from here, [Baidu Drive](https://pan.baidu.com/s/1pbRUmRg_Y69KRNEuKZParQ), and the password is <b>1s5t</b>.
Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

## Process raw data for METR-LA and PEMS-BAY

### Create data directories
```
mkdir -p data/{METR-LA,PEMS-BAY}
```

### METR-LA
```
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
```
### PEMS-BAY
```
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Train Commands for PEMS08 

```
python train.py --device cuda:0
```

## Test Commands for PEMS08 

```
python test.py --checkpoint garage8/PEMS08_epoch_158_16.75.pth --batch_size 1 --device cuda:0
```

## Citation

If you find this useful, please cite our paper: "STLGRU: Spatio-temporal lightweight graph GRU for traffic flow prediction"
```

@inproceedings{bhaumik2024stlgru,
  title={STLGRU: Spatio-temporal lightweight graph GRU for traffic flow prediction},
  author={Bhaumik, Kishor Kumar and Niloy, Fahim Faisal and Mahmud, Saif and Woo, Simon S},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={288--299},
  year={2024},
  organization={Springer}
}
```


