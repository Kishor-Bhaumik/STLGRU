## Data Preparation
Download the dataset(PEMS03, PEMS04, PEMS07, PEMS08) from here, [Baidu Drive](https://pan.baidu.com/s/1pbRUmRg_Y69KRNEuKZParQ), and the password is <b>1s5t</b>.

## Train Commands

```
python train.py --device cuda:0
```

## Test Commands

```
python test.py --checkpoint garage8/PEMS08_epoch_158_16.75.pth --batch_size 1 --device cuda:0
```
