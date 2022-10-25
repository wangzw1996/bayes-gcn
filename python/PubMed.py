from __future__ import division

import time
import argparse
import os.path as osp
import torch
from torch import tensor
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import BiGCN_layerspar
import numpy as np
import torch_geometric.transforms as T
from utils import load_data, accuracy_mrun_np, normalize_torch,uncertainty_mrun
import pandas as pd
from numpy import *
from torch_geometric.datasets import Reddit, Flickr


seed = 20
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    

#print(torch.cuda.get_device_name(1))

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PubMed')
dataset = Planetoid(path,'PubMed',transform=T.NormalizeFeatures())
data = dataset[0]




labels_np = data.y.cpu().numpy().astype(np.int32)
idx_test_np = data.test_mask.cpu().numpy()


def train(model, optimizer, data):
   
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    pred = out[data.train_mask].max(1)[1]
        
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    loss.backward()

    optimizer.step()
    return acc


def evaluate(model, data,num_run,bi_second):
    model.eval()
    #print('test')
    with torch.no_grad():
         
        acc_train= [None]*num_run
        acc_test= [None]*num_run
        acc_val= [None]*num_run
        val_loss=[None]*num_run
        outs_opt = [None]*num_run
        outs = {}
        logits = model.inference1(data,0,bi_second)
        
        
        for j in range(num_run): 
               
          logits2 = model.inference2(logits,0,data.edge_index,bi_second) 
          outs_opt[j] = logits2.cpu().data.numpy()
          
          
          
          for key in ['train', 'val', 'test']:
              mask = data['{}_mask'.format(key)]
              val_loss = F.nll_loss(logits2[data['val_mask']], data.y[data['val_mask']]).item()
              
              pred_train = logits2[data['train_mask']].max(1)[1]
              pred_val = logits2[data['val_mask']].max(1)[1]
              pred_test = logits2[data['test_mask']].max(1)[1]
              
              acc_train = pred_train.eq(data.y[data['train_mask']]).sum().item() / data['train_mask'].sum().item()
              acc_val = pred_val.eq(data.y[data['val_mask']]).sum().item() / data['val_mask'].sum().item()
              acc_test = pred_test.eq(data.y[data['test_mask']]).sum().item() / data['test_mask'].sum().item()
              
        outs_opt= np.stack(outs_opt)
        c_opt=torch.tensor(outs_opt)
        labels_opt=torch.tensor(labels_np)
        idx_test=torch.tensor(idx_test_np)
        pavpu_opt=uncertainty_mrun(c_opt, labels_opt, idx_test)
        for key in ['train', 'val', 'test']:
          outs_train = mean(acc_train)
          outs_test = mean(acc_test)
          outs_val = mean(acc_val)
    
    return pavpu_opt,outs_train,outs_test,outs_val,val_loss
    
    

        
           



def run(exp_name, data, model, runs, epochs, lr, weight_decay, early_stopping, device,bi_first,bi_second,dropout,dropout2,num_run):
    val_losses, accs, durations = [], [], []
    for run_num in range(runs):
        #begin=time.time()
        #x=data.x.to(device)
        #edge=data.edge_index.to(device)
        #end=time.time()
        #print(end-begin)        
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_acc = 0
        best_val_loss = 0
        train_acc = 0
        val_acc = 0
        test_acc = 0
        val_loss_history = []
        epoch_count = -1

        for epoch in range(1, epochs + 1):
            # print("epochs:",epoch)
            train(model, optimizer, data)
            pavpu,eval_train,eval_test,eval_val,val_loss = evaluate(model, data,num_run,bi_second)
           

            if eval_val > best_val_acc:
                best_val_loss = val_loss
                test_acc = eval_test
                train_acc=eval_train
                
                val_acc = eval_val
                epoch_count = epoch
                pavpu_opt=pavpu
                #torch.save(model.state_dict(),"/mnt/ccnas2/bdp/zw4520/gcn/pumbed.pkl")
                # torch.save(model.state_dict(), 'gcn_cora.pkl')
               

            val_loss_history.append(val_loss)
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

            if runs == 1:
                print(epoch, "train: {:.4f}".format( train_acc),
                      "val: {:.4f}".format(val_acc),
                      "test: {:.4f}".format(test_acc))
               

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print("Run: {:d}, dropout_rate: {:.4f},train_acc: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}".format(run_num+1, dropout,train_acc, val_acc, test_acc))
        print('pavpu')
        print(pavpu_opt)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Experiment:', exp_name)
    print('Binarized: {:.0f},Precision sparsity: {:.0f},Dropout rate: {:.3f},Dropout mode:{}, samples : {:.0f}, Test Accuracy: {:.4f}, std: {:.4f}'.
          format(bi_first,bi_second, dropout,dropout2,num_run, acc.mean().item(), acc.std().item(),
                 ))
                 
    return    acc.mean()  
    
    
def run2(exp_name, data, model, runs, epochs, lr, weight_decay, early_stopping, device,bi_first,bi_second,dropout,num_run):
    val_losses, accs, durations = [], [], []
    for run_num in range(runs):
        #begin=time.time()
        #x=data.x.to(device)
        #edge=data.edge_index.to(device)
        #end=time.time()
        #print(end-begin)        
        data = data.to(device)
        model.to(device)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_acc = 0
        best_val_loss = 0
        train_acc = 0
        val_acc = 0
        test_acc = 0
        val_loss_history = []
        epoch_count = -1

        for epoch in range(1, epochs + 1):
            # print("epochs:",epoch)
            
            pavpu,eval_train,eval_test,eval_val,val_loss = evaluate(model, data,num_run,bi_second)
           

            if eval_val > best_val_acc:
                best_val_loss = val_loss
                test_acc = eval_test
                
                val_acc = eval_val
                epoch_count = epoch
                pavpu_opt=pavpu
                
                

            val_loss_history.append(val_loss)
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

            if runs == 1:
                print(epoch,
                      "val: {:.4f}".format(val_acc),
                      "test: {:.4f}".format(test_acc))
               

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
        print("Run: {:d}, dropout_rate: {:.4f}, val_acc: {:.4f}, test_acc: {:.4f}".format(run_num+1, dropout, val_acc, test_acc))
        print('pavpu')
        print(pavpu_opt)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Experiment:', exp_name)
    print('Binarized: {:.0f},Precision sparsity: {:.0f},Dropout rate: {:.3f},samples : {:.0f}, Test Accuracy: {:.4f}, std: {:.4f}'.
          format(bi_first,bi_second, dropout,num_run, acc.mean().item(), acc.std().item(),
                 ))
                 
    return    acc.mean()  
    
    
    
    
            
            
            
       


def main():



    import sys
    class Logger(object):
      def __init__(self,logFile ='Default.log'):
         self.terminal = sys.stdout
         self.log =open(logFile,'a')
         
      def write(self,message):
         self.terminal.write(message)
         self.log.write(message)
         
      def flush(self):
          pass
    sys.stdout=Logger('/mnt/ccnas2/bdp/zw4520/gcn/pubmed_8.log')
    import logging
    
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.DEBUG)
    handler = logging.FileHandler('/mnt/ccnas2/bdp/zw4520/gcn/pubmed1_8.log')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
 
    logger.info("")
    logger.debug("")
    logger.warning("")
    logger.info("")
    
    
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='1', type=str, help='gpu id')
    parser.add_argument('--exp_name', default='default_exp_name', type=str)
    parser.add_argument('--dataset', type=str, default='PubMed')  # Cora/CiteSeer/PubMed
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4
    parser.add_argument('--early_stopping', type=int, default=0)  # 100
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)  # 0.5
    args = parser.parse_args()
    print(args)
    
    
    sample_num=[5,10,15,20] 
    dropout_num=[0.125,0.25,0.5]
    dropout_mode=['output','input']
    device = torch.device('cuda:1')
    
    
    print("Size of train set:", data.train_mask.sum().item())
    print("Size of val set:", data.val_mask.sum().item())
    print("Size of test set:", data.test_mask.sum().item())
    print("Num classes:", dataset.num_classes)
    print("Num features:", dataset.num_features)


    
    for dropout in dropout_num:
       best_accuracy=0
       best_dropout =0
       model = BiGCN_layerspar(dataset.num_features, args.hidden, dataset.num_classes, args.layers,0,dropout,'output',False,False)
       accuracy = run(args.exp_name, dataset[0], model, args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, device,False,False,dropout,'output',20)
       
       if best_accuracy < accuracy:
          best_accuracy=accuracy
          best_dropout= dropout
       print(dropout)
       
       
    for dropout in dropout_num:
     for dropout2 in dropout_mode :
       best_accuracy=0
       best_dropout =0
       model = BiGCN_layerspar(dataset.num_features, args.hidden, dataset.num_classes, args.layers,0,dropout,dropout2,True,False)
       accuracy = run(args.exp_name, dataset[0], model, args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, device,True,False,dropout,dropout2,20)
       
       if best_accuracy < accuracy:
          best_accuracy=accuracy
          best_dropout= dropout      
          
                              
        
    for sample in sample_num:
      for dropout in dropout_num:
       for dropout2 in dropout_mode :
        #model.load_state_dict(torch.load("/mnt/ccnas2/bdp/zw4520/gcn/pumbed.pkl"))
        model = BiGCN_layerspar(dataset.num_features, args.hidden, dataset.num_classes, args.layers,0,dropout,'input',True,True)
        accuracy = run(args.exp_name, dataset[0], model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, device,True,True,dropout,dropout2,sample)
       


       
    
       
       
       
      
    
    
         
       
      


if __name__ == '__main__':
    main()
