import argparse
import logging
import os
import pandas as pd
import numpy as np

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', help='full compositionality estimates '
                        'dataframe output by compos_estimates.py')
    parser.add_argument('gold_df', help='gold standard dataframe pickle '
                        'assuming columns `target`, `cpd`, `head`, `modif`')
    parser.add_argument('out_file', help='path where to write output pickle')

    args = parser.parse_args()
    
    in_file = args.in_file
    gold_file = args.gold_df
    out_file = args.out_file
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logging.info(f'Evaluating compositionality estimates: {in_file}')
    logging.info(f'Gold standard data: {gold_file}')
    
    master_df = pd.read_pickle(in_file)
    gold = pd.read_pickle(gold_file)
    print(gold.head(5))
    #assert list(master_df.columns) == gold['target'].to_list()
    
    gold = gold[['target', 'cpd', 'head', 'modif']]
    gold = gold.set_index('target')
    
    # Compute correlations for each prediction target
    corr_dfs = []
    preds = ['cpd', 'head', 'modif']
    for pred in preds:
        corr_df = master_df.T.corrwith(gold[pred], method='spearman')
        corr_dfs.append(corr_df)
    
    # Concatenate and dump final DF
    master_corr_df = pd.concat(corr_dfs, axis=1, keys=preds)
    master_corr_df.to_pickle(out_file)
    
    logging.info(f'Correlations dataframe written: {out_file}')
    
if __name__ == '__main__':
    main()
