#!/usr/bin/env python3

# Copyright (C) 2020 <Henry Turner, Giulio Lovisotto, Ivan Martinovic>



import pandas as pd 
import numpy as np
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('version', type=int, choices=[1, 2])
    parser.add_argument('set', type=str, choices=['dev', 'test'])
    parser.add_argument('out_dir')

    args = parser.parse_args()

    # acquire meta file from input dir

    meta_file = args.input_dir + '/' + 'vox' + str(args.version) + '_meta.csv'
    outfile = os.path.join(args.out_dir, 'spk2gender')
    if os.path.isfile(outfile):
        sys.exit()

    seperator = '\t' if args.version == 1 else ','
    id_column = 'VoxCeleb' + str(args.version) + ' ID'
    df = pd.read_csv(meta_file, sep=seperator)

    # Strip all the white spaces in column names and in the data
    cols = [col.strip() for col in df.columns]
    df.columns = cols
    df[cols] = df[cols].apply(lambda x: x.str.strip())

    # Remove the ones we do not need
    df = df[df['Set'] == args.set]

    #Select only ID and gender rows
    df = df[[id_column, 'Gender']]

    # The VoxCeleb2 metadata contains two entries that were removed for some reason in the final dataset
    # These are id4170 and id 5348.
    # Removing these also produces the correct count of speakers for the test set, and no audio is downloadable for these
    # We add a short function here to remove these

    if args.version == 2 and args.set == 'test':
        df = df[~((df[id_column] == 'id04170') | (df[id_column] == 'id05348'))]

    # write to file without headers
    outfile = os.path.join(args.out_dir, 'spk2gender')
    df.to_csv(outfile, sep=' ', header=False, index=False)
