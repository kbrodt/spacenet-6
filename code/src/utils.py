import pandas as pd
import skimage
import skimage.io
import numpy as np


def split_df(df, args):
    df = df.sample(frac=1, random_state=args.seed).reset_index()
    df['fold'] = np.arange(len(df)) % args.n_folds
    
    return df[df.fold != args.fold].reset_index(), df[df.fold == args.fold].reset_index()


def get_data_groups(path, args):
    train = pd.read_csv(path)
    train['is_test'] = False
    
    dev = pd.read_csv(path.parent / path.name.replace('train', 'valid'))
    dev['is_test'] = False
    train = train.append(dev)
    train, dev = split_df(train, args)
    
    if args.ft:
        train = pd.concat([train, dev]).reset_index(drop=True)
    
    if args.pl is not None:
        test = pd.read_csv(args.pl)
        test = test[test.has_mask].copy()
        test['is_test'] = True
        train = pd.concat([train, test]).reset_index(drop=True)
    
    return train, dev


def read_img_ski(path):
    return skimage.io.imread(str(path))
