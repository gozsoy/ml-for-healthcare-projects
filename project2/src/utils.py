import os
import random
import argparse

import ray
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf


def set_seeds(cfg):
    seed = cfg['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    return


def load_data(cfg, split_type, cached=True):

    if cached:
        file_path = os.path.join(cfg['data_dir'], split_type+'_cleaned.pickle')
        if os.path.isfile(file_path):
            df = pd.read_pickle(file_path)
        else:
            df = preprocess_raw(cfg=cfg, split_type=split_type)
    else:
        df = preprocess_raw(cfg=cfg, split_type=split_type)

    print(f'{split_type} data loaded.')
    return df


def preprocess_raw(cfg, split_type):
    print(f'Preprocessing raw {split_type} data from scratch...')
    read_dir = os.path.join(cfg['data_dir'], split_type+'.txt')
    write_dir = os.path.join(cfg['data_dir'], split_type+'_cleaned.pickle')

    with open(read_dir) as f:
        lines = f.readlines()

    label_set = ['OBJECTIVE', 'BACKGROUND',
                 'METHODS', 'RESULTS', 'CONCLUSIONS']
    label_text_tuples = []

    for line in lines:
        lbl, _, txt = line.partition('\t')
       
        if lbl in label_set:
            label_text_tuples.append((lbl, txt))

    df = pd.DataFrame(label_text_tuples, columns=['label', 'text'])

    nlp = spacy.load('en_core_web_sm')

    @ray.remote
    def preprocess_f(row, cfg):
        processed_token_list = []

        def remove_f(cfg, t, type):
            if type == 'stop':
                if cfg['remove_stop']:
                    return not t.is_stop
                else:
                    return True
            else:  # punc
                if cfg['remove_punc']:
                    return not t.is_punct
                else:
                    return True

        for t in nlp(row):
            if remove_f(cfg, t, 'stop') and remove_f(cfg, t, 'punc') \
                                        and not t.is_space:
                if len(t) >= cfg['min_token_len'] and \
                   len(t) <= cfg['max_token_len']:
                    if t.like_num and cfg['number2hashsign']:
                        processed_token_list.append('#')
                    else:
                        if cfg['lemmatize']:
                            processed_token_list.append(t.lemma_.lower())
                        else:
                            processed_token_list.append(t.lower())
                            
        return " ".join(processed_token_list)

    ray.init()
    df['cleaned_text'] = ray.get([preprocess_f.remote(row, cfg) 
                                  for row in df['text']])
    ray.shutdown()

    df = df[['label', 'cleaned_text']]
    df.to_pickle(write_dir, protocol=4)
    
    return df


def get_argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default='../data/PubMed_200k_RCT')
    parser.add_argument("--checkpoint_dir", type=str,
                        default='../checkpoints')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--cached", type=bool, default=True)
    parser.add_argument("--vector_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--w2v_epochs", type=int, default=5)
    parser.add_argument("--finetune_bert", type=bool, default=False)
    parser.add_argument("--retrain_task", type=bool,
                        default=False)  # not functional yet
    # below are preprocessing options                    
    parser.add_argument("--remove_stop", type=bool, default=True)
    parser.add_argument("--remove_punc", type=bool, default=True)
    parser.add_argument("--number2hashsign", type=bool, default=False)
    parser.add_argument("--min_token_len", type=int, default=1)
    parser.add_argument("--max_token_len", type=int, default=15)
    parser.add_argument("--lemmatize", type=bool, default=True)
    
    return parser