import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from miditok import REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
import time
import gc
from tqdm import tqdm
import copy
from collections import defaultdict
import os
import logging
import pickle
from config import *
from utils import train_one_epoch,valid_one_epoch
from data import MidiDataset
from model import Transformer

if __name__=="__main__":
    # prepping directories
    os.makedirs(MODEL_DIR,exist_ok=True)
    os.makedirs(LOG_DIR,exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG (you can adjust as needed)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Define the log message format
        filename=LOG_DIR+'/app_{}.log'.format(SUFFIX),  # Specify the log file name
        filemode='w'  # Set the file mode to write (overwrite if the file already exists)
    )
    logging.info("loading files...")
    with open(ENCODING_DIR+"/tokenizer.pkl", 'rb') as file:
        tokenizer = pickle.load(file)
    logging.info("tokenizer length {}".format(tokenizer.len))
    df = pd.read_csv(ENCODING_DIR+'/df.csv')
    logging.info("dataset size : {}".format(df.shape[0]))

    # data prep
    train_df = df.loc[:int(len(df)*TRAIN_TEST_RATIO),:]
    val_df = df.loc[int(len(df)*TRAIN_TEST_RATIO):,:]

    train_data = MidiDataset(train_df)
    eval_data = MidiDataset(val_df)
    train_dataloader = torch.utils.data.DataLoader(train_data,\
                            batch_size=BATCH_SIZE,\
                            shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_data,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
    # model prep
    model = Transformer(vocab_size = VOCAB_SIZE,
                        num_embed = EMBED_DIM,
                        block_size = SEQ_LEN,
                        num_heads = TRANSFORMER_HEADS,
                        num_layers = TRANSFORMER_LAYERS)
    model.to(DEVICE)
    param_optimizer = model.parameters()
    optimizer = torch.optim.AdamW(param_optimizer, lr=LR)

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)


    for epoch in range(1,EPOCHS+1):
        gc.collect()
        train_loss = train_one_epoch(model,optimizer,train_dataloader,DEVICE,epoch)

        val_loss = valid_one_epoch(model,optimizer,eval_dataloader,DEVICE,epoch)

        history["TrainLoss"].append(train_loss)
        history["ValLoss"].append(val_loss)


        logging.debug("Epoch: {} TL: {} VL: {}".format(epoch,train_loss,val_loss))
        if val_loss < best_epoch_loss:
            logging.debug(f"Validation Loss Improved ({best_epoch_loss} ---> {val_loss})")
            best_epoch_loss = val_loss
            
            best_model_wts = copy.deepcopy(model.state_dict())
            
            PATH ="model.pt"
            torch.save(model.state_dict(), os.path.join(MODEL_DIR,PATH))
            
            logging.debug("Model Saved")
    end = time.time()
    time_elapsed = end - start
    logging.debug('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logging.debug("Best Loss: {:.4f}".format(best_epoch_loss))
    PATH ="lst_model.pt"
    torch.save(model.state_dict(), os.path.join(MODEL_DIR,PATH))

