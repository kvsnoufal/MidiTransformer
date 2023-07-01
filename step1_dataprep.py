from miditok import REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from glob import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *

if __name__ == "__main__":
    files = glob(MIDI_DIR+"/*/*/*/*.mid")
    print("All Midi files: ", len(files))

    SAVE_FOLDER = ENCODING_DIR
    os.makedirs(SAVE_FOLDER,exist_ok=True)
    tokenizer = REMIPlus()

    print("converting to numpy and saving...")

    df = []
    for f in tqdm(files,total=len(files)):
        # break
    # os.path.basename(f)
        try:
            midi = MidiFile(f)
            tokens = list(tokenizer(midi))
            tokens = np.array(tokens)
            # tokens.shape
            savefilename = os.path.basename(f)[:-4]+".npy"
            # savefilename
            savefilename = os.path.join(SAVE_FOLDER,savefilename)
            # savefilename
            with open(savefilename,'wb') as f_:
                np.save(f_,tokens)
            df.append([os.path.basename(savefilename),tokens.shape[0]])
        except:
            continue


    df = pd.DataFrame(df)
    print(df.head())
    print()
    df = df.rename(columns={0:"fname",1:"tokenLength"})

    print(df.describe())
    print()
    df.to_csv(os.path.join(SAVE_FOLDER,"df.csv"),index=None)
    print("saving tokenizer...")
    tokenizer.save_params(SAVE_FOLDER+"tokenizer.json")
    print("Vocab size: ",tokenizer.len)