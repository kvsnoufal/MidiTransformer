from tqdm import tqdm
import gc
import torch

def train_one_epoch(model, optimizer,  dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        x = data['x'].to(device)
        y = data['y'].to(device)
        
        batch_size = x.size(0)
        
        logits,loss = model.forward(x,y)
        
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
            
                
        running_loss += loss.item()
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss  
def valid_one_epoch(model, optimizer,  dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for step, data in bar:
            x = data['x'].to(device)
            y = data['y'].to(device)
            
            batch_size = x.size(0)
            
            logits,loss = model.forward(x,y)
            
                    
            running_loss += loss.item()
            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            
            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                            LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss