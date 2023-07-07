import argparse
import logging
import os
import pandas as pd


def process_direct_estimates(directories, collapse_pooling=True):
    """Processes individual output files w/ direct compositionality estimates.
    
    Args:
        directories: One or more input directories containing files output
            by compos_bert.py.
        collapse_pooling: Assumes that a single pooling method is applied
            at all levels (layers, subwords, sequences) and simplifies config
            descriptions accordingly.
    
    Returns:
        Dataframe with direct estimates for all compounds.
    """
    
    master_dfs = []

    for directory in directories:

        master_df = pd.DataFrame()

        targets = sorted(os.listdir(directory), key=str.casefold)
        targets = [t[:-4] for t in targets]
        # Filter out temp files etc.
        targets = [t for t in targets if not (t.startswith('.') or
                                              t.endswith('_'))]
        existing_idx = []
        for i, target in enumerate(targets):
            df = pd.read_pickle(os.path.join(directory, target + '.pkl'))
            target = target.lower()

            if collapse_pooling:
                # Clean up column names
                df['type'] = df['type'].map({'token': 'token',
                                             'type-avg': 'type', 
                                             'type-sum': 'type'})

            # Merge combo columns into single string
            config_cols = ['layer_combo', 'layer_pooling', 
                           'min_seq_len', 'n_seqs', 'emb1', 'emb2', 'type']

            joiner = lambda x: '_'.join(x.astype(str))
            df['combo'] = df[config_cols].apply(joiner, axis=1)

            # Trim down target word DF
            df = df.set_index('combo')[['value']]
            df = df.rename(columns={'value': target})

            # verify indexes are in same order
            if len(existing_idx):
                assert df.index.values.tolist() == existing_idx
            else:
                existing_idx = df.index.values.tolist()

            # print(df.head(5))
            # exit()
            # Merge target word DF onto master DF
            # master_df = master_df.merge(df,
            #                             left_index=True,
            #                             right_index=True,
            #                             how='outer')
            master_df = pd.concat((master_df, df), axis=1)

            assert master_df.shape[0] == df.shape[0]
            assert master_df[target].isna().sum() == 0
            # master_df = master_df.drop_duplicates()

        master_dfs.append(master_df)
        logging.info(f'Processed direct estimates: {directory}')
        
    master_df = pd.concat(master_dfs)
    print(master_df.info())
    print(master_df.head(6))
    return master_df


def get_composite_estimates(master_df):
    """Computes composite compositionality estimates from direct estimates.
    
    Args:
        master_df: Dataframe with all direct estimates.
        
    Returns:
        Dataframe with all direct + composite estimates.
    """
    
    # Extract core parameter combinations (without pairs of embedding types)
    core_combos = master_df.index.to_list()
    core_combos = [c.split('_') for c in core_combos]
    core_combos = ['_'.join(c[:4] + c[-1:]) for c in core_combos]
    core_combos = list(set(core_combos))
    
    # Compute composite estimates
    others = ['mwe', 'context', 'cls'] # values to combine with head/modif

    # todo rewrite for magnitude
    funcs = {'add': lambda x: x[0] + x[1],
             'mult': lambda x: x[0] * x[1],
             'comb': lambda x: (x[0] + x[1]) + (x[0] * x[1])}

    for i, core_combo in enumerate(core_combos):
        core_combo = core_combo.split('_')

        for other in others:
            head_score = f'head_{other}'
            modif_score = f'modif_{other}'
            head_score = core_combo[:4] + [head_score] + core_combo[-1:]
            modif_score = core_combo[:4] + [modif_score] + core_combo[-1:]
            head_score = '_'.join(head_score)
            modif_score = '_'.join(modif_score)

            target_rows = (master_df.loc[head_score], 
                           master_df.loc[modif_score])

            for func in funcs:
                new_combo = core_combo[:4] + [f'{other}_{func}'] + core_combo[-1:]
                new_combo = '_'.join(new_combo)

                new_row = funcs[func](target_rows).drop_duplicates()
                assert len(new_row) == 1

                master_df.loc[new_combo] = new_row.iloc[0]
                
    return master_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('in_dirs', help='one or more directories containing '
                        'files output by compos_bert.py', nargs='+')
    parser.add_argument('out_file', help='path where to write output pickle')

    args = parser.parse_args()
    
    in_dirs = args.in_dirs
    out_file = args.out_file
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    logging.info(f'Processing direct estimates from {len(in_dirs)} directories.')
    master_df = process_direct_estimates(in_dirs)
    
    logging.info('Computing composite estimates.')
    master_df = get_composite_estimates(master_df)
    
    master_df.to_pickle(out_file)
    logging.info(f'Output written: {out_file}')
    

if __name__ == '__main__':
    main()
