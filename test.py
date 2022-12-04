import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import os
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--garage',type=str,default='./garage8',help='garage path')
parser.add_argument('--device',type=str,default='cuda:4',help='')
parser.add_argument('--batch_size',type=int,default=128,help='batch size')
parser.add_argument('--data',type=str,default='data/PEMS08',help='data path')
parser.add_argument('--adjdata',type=str,default='data/PEMS08/adj_pems08.pkl',help='adj data path')
parser.add_argument('--num_nodes',type=int,default=170,help='number of nodes')    #7 :-> 883   4:-> 307  3:- 358   8:-> 170

parser.add_argument('--out_length',type=int,default=12,help='Forecast sequence length')

parser.add_argument('--n_hid',type=int,default=64,help='')
parser.add_argument('--input_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--seq_length',type=int,default=12,help='Input sequence length')


parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate')
parser.add_argument('--epochs',type=int,default=200,help='') # 200
parser.add_argument('--print_every',type=int,default=100,help='Training print')
parser.add_argument('--save',type=str,default='./garage/PEMS08',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--max_update_factor',type=int,default=1,help='max update factor')
parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--checkpoint',type=str,default="False", help='')

args = parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(args.seed)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def test_model(engine,checkpoint,dataloader,device):
    valid_loss = []
    valid_mae = []
    valid_mape = []
    valid_rmse = []

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx
        testy = torch.Tensor(y).to(device)
        testy = testy
        metrics = engine.test(testx, testy[:,:,:,0],checkpoint)
        valid_loss.append(metrics[0])
        valid_mae.append(metrics[1])
        valid_mape.append(metrics[2])
        valid_rmse.append(metrics[3])
    
    mvalid_loss = np.mean(valid_loss)
    mvalid_mae = np.mean(valid_mae)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    print(" ")
    print(checkpoint, " loaded successfully")
    log = 'test Loss: {:.4f}, test MAE: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}'
    print(log.format( mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse))

    return mvalid_mae




def main2():
    device = torch.device(args.device)
    adj = util.load_adj(args.adjdata)
    if 'PEMS08' not in args.adjdata: adj =adj[2]
    adj= torch.from_numpy(adj.astype(np.float32)).to(device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']    
    global_train_steps = dataloader['train_loader'].num_batch

    engine = trainer(scaler, args, adj, global_train_steps , device)
    files = Path(args.garage).glob('*')
    best_model = {}
    for c, checkpoint in enumerate(files):
        mae = test_model(engine,checkpoint,dataloader,device)
        best_model[mae]= checkpoint

        #if c==2: break

    lowest_mae = min(best_model)
    print(" ")
    print("best model :---->",best_model[lowest_mae], " ", lowest_mae)
    mae = test_model(engine,best_model[lowest_mae],dataloader,device)


def main():
    device = torch.device(args.device)
    adj = util.load_adj(args.adjdata)
    if 'PEMS08' not in args.adjdata: adj =adj[2]
    adj= torch.from_numpy(adj.astype(np.float32)).to(device)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']    
    global_train_steps = dataloader['train_loader'].num_batch

    print(" ")
    print(args.checkpoint)
    
    engine = trainer(scaler, args, adj, global_train_steps , device)
    
    macs , params = engine.check_parameter((12,170,1))
    print(" ")
    print(macs, params)
    print(" ")


    valid_loss = []
    valid_mae = []
    valid_mape = []
    valid_rmse = []

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx
        testy = torch.Tensor(y).to(device)
        testy = testy
        metrics = engine.test(testx, testy[:,:,:,0],args.checkpoint)
        valid_loss.append(metrics[0])
        valid_mae.append(metrics[1])
        valid_mape.append(metrics[2])
        valid_rmse.append(metrics[3])
    
    mvalid_loss = np.mean(valid_loss)
    mvalid_mae = np.mean(valid_mae)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)

    log = 'test Loss: {:.4f}, test MAE: {:.4f}, test MAPE: {:.4f}, test RMSE: {:.4f}'
    print(log.format( mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse))

if __name__ == "__main__":
    if args.checkpoint != "False":
        main()
    else : main2()



# if __name__ == "__main__":
#     main()
