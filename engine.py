import torch
import torch.optim as optim
from model import model
import util
import pdb

class trainer():
    def __init__(self, scaler, args, adj, global_train_steps, device):
        lr_new = util.lr_new(args, global_train_steps)
        self.model = model(args,adj).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda =lambda num_update:(lr_new.update(num_update)/args.learning_rate))
        self.loss = util.huber_loss
        self.scaler = scaler
        self.args = args
        self.adj =adj
        self.device = device


    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #pdb.set_trace()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, rho=1, null_val=0.0)
        loss.backward()
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, rho=1, null_val=0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def test(self, input, real_val,checkpoint):
        self.model.load_state_dict(torch.load(checkpoint,map_location=self.device))

        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, rho=1, null_val=0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    
    def check_parameter(self, rand):
        macs, params = get_model_complexity_info(self.model, rand , as_strings=True, print_per_layer_stat=True, verbose=True)
        return macs , params