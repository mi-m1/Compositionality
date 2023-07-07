import argparse
import html
import logging
import numpy as np
import os
import pandas as pd
import re
import string

from collections import defaultdict
from bert_utils import *


def read_examples(target, directory=''):
    """Reads sentences containing target MWE from a file, 1 sentence per line.
    
    Args:
        target: Target MWE. It is assumed that input file is of the same name,
            with underscores instead of spaces and a .txt extension.
        directory: Directory containing the target file.
    
    Returns:
        A list where each item is a sentence, rstripped and HTML-unescaped.
    """

    target_file = target.replace(' ', '_') + '.txt'
    target_file = os.path.join(directory, target_file)

    with open(target_file, 'r') as f:
        examples = f.readlines()
    examples = [html.unescape(l.rstrip()) for l in examples]
    
    return examples


def cosine(x, y, distance=False):
    """Computes cosine similarity or distance for two tensors."""
    
    cosine = torch.nn.CosineSimilarity(dim=0)
    out = cosine(x, y).item()
    if distance:
        out = 1 - out

    return out


def get_layer_combos(min_layer=1, max_layer=12, how='win', custom_layers=None):
    """Generates layer combinations to be used for embedding generation.
    
    Layer indices are in range [0, 12], as per the Huggingface Transformers
    output of BERT hidden states, where 0 corresponds to the initial embedding
    input, and 12 corresponds to the output of the last BERT layer.
    
    Args:
        min_layer: First layer.
        max_layer: Final layer.
        how: 'full' produces all unique combinations of the included layers;
            'win' produces all contiguous combinations (sliding window);
            'debug' produces a single combination -- the longest. The
            number of layers in the combinations is in range [1, n_layers].
        custom_layers: Custom list of layers to be included, superseding the
            contiguous range defined by [min_layer, max_layer].
    
    Returns:
        A list of tuples of layer indices.
    """
    
    if custom_layers is not None:
        all_layers = custom_layers
    else:
        all_layers = list(range(min_layer, max_layer+1))
        
    n_all_layers = len(all_layers)

    if how == 'debug':
        combos = [all_layers]
    else:
        combos = []

        for n_layers in range(1, n_all_layers+1):
            if how == 'full':
                combos += list(combinations(all_layers, n_layers))
            elif how == 'win':
                for i in range(n_all_layers - n_layers + 1):
                    j = i + n_layers
                    combos.append(all_layers[i:j])

    return combos


def apply_prefilter(target, seq):
    """Applies regex-based filters on corpus sequence."""
    
    punct = r"[{}]".format("'!\"#$%&\\\\'()*+,-./:;<=>?@[\\]^_`{|}~'")
    re_punct = re.compile(r"{}\s?{}|{}\s?{}".format(punct, target, target, punct), flags=re.I)
    re_position = re.compile(f'^{target}|{target}$', flags=re.I)

    approved = ['none']
    
    if not re.search(re_position, seq):
        approved.append('position')
    if not re.search(re_punct, seq):
        approved.append('punct')
    if 'punct' in approved and 'position' in approved:
        approved.append('all')
        
    return approved


def filter_examples(target, examples, prefilters, max_n_seq, min_seq_lens):
    """Filters corpus examples based on preprocessing evaluation variables.
    
    Args:
        target: Target MWE.
        examples: List of examples to be filtered.
        prefilters: List of regext-based filters to be applied, out of 
            ['none', 'punct', 'position', 'all']. Currently only relevant
            if using ['none'], to skip other filters; otherwise full
            list is assumed.
        max_n_seq: Highest tested number of sequences, used as cutoff.
        min_seq_lens: List of minimum sequence lengths that are tested.
        
    Returns:
        A dict of dicts of sequence indices, structured as follows:
        {prefilter1: {min_seq_len1: [seq_idx1, ...], ...}, ...}
    """
    
    filtered = {}
    
    # Check if prefiltering criteria should be skipped
    if len(prefilters) == 1 and prefilters[0] == 'none':
        cutoff_cat = 'none'
    else:
        cutoff_cat = 'all'
    
    # Generates all pre-removal criteria combinations;
    # same even if skipping criteria to conserve data structure
    for prefilter in prefilters:
        filtered[prefilter] = {}
        for min_seq_len in min_seq_lens:
            filtered[prefilter][min_seq_len] = []


    cutoff_seq_len = max(min_seq_lens) # highest criterion

    # Iterates over corpus sequences, validates each against all criteria combos,
    # and assigns each to all lists of sequences for which criteria combo is met.
    # Cutoff: maximum number of tested sequences reached for most stringent combo.
    for i, seq in enumerate(examples):
        if cutoff_cat == 'all':
            approved_cats = apply_prefilter(target, seq)
        else:  # skipping preapproval
            approved_cats = ['none']
            
        # Exclude sequences that would be too long for BERT.
        # TODO: streamline BERT tokenization, done twice: here and when modeling.
        bert_seq_len = len(tokenizer.encode(seq))
        if bert_seq_len > 512:
            continue
        
        seq_len = len(seq.split())

        for cat in approved_cats:
            for min_seq_len in min_seq_lens:
                if seq_len >= min_seq_len:
                    filtered[cat][min_seq_len].append(i)

        if len(filtered[cutoff_cat][cutoff_seq_len]) >= max_n_seq:
            break
            
    # Limit lists to required number of examples
    for cat in filtered:
        for min_seq_len in filtered[cat]:
            filtered[cat][min_seq_len] = filtered[cat][min_seq_len][:max_n_seq]
            
    return filtered


def get_unique_filtered_seqs(filtered_seqs):
    """Retrieves the union of unique prefiltered sequence indices."""
    
    unique_seqs = [seq for prefilter in filtered_seqs
                       for min_seq_len in filtered_seqs[prefilter]
                       for seq in filtered_seqs[prefilter][min_seq_len]]
    
    unique_seqs = list(set(unique_seqs))
    
    return unique_seqs


def update_all_indices_in_seq(modif_ids, head_ids, seq_ids, indices_to_update):
    """Updates master dictionary with indices for target tokens in a sequence.
    
    Args:
        modif_ids: List of modifier token IDs.
        head_ids: List of head token IDs.
        seq_ids: List of token IDs for the full tokenized sequence.
        indices_to_update: Dictionary containing indices for different target
            tokens ('modif', 'head', 'mwe', 'context', 'cls'), for the whole
            batch of sequences.
            
    Returns:
        None; indices_to_update is updated.
    """
    
    # Get modifier and head indices in the sequence
    modif_indices, head_indices = get_indices_in_seq(modif_ids=modif_ids,
                                                     head_ids=head_ids,
                                                     seq_ids=seq_ids)
    
    # Get MWE (modifier + head) indices in the sequence
    n_mwes_in_seq = len(modif_indices)
    mwe_indices = [modif_indices[i] + head_indices[i] 
                   for i in range(n_mwes_in_seq)]
    
    # Get context indices (all tokens apart from MWE, CLS, SEP)
    n_tokens_in_seq = seq_ids.size(dim=1)
    context_indices = []
    for mwe_in_seq in mwe_indices:
        context_in_seq = [i for i in range(1, n_tokens_in_seq-1)
                            if i not in mwe_in_seq]
        context_indices.append(context_in_seq)
    
    # CLS
    cls_indices = [[0] for i in range(n_mwes_in_seq)]
    
    # Update master dictionary
    indices_to_update['modif'].append(modif_indices)
    indices_to_update['head'].append(head_indices)
    indices_to_update['mwe'].append(mwe_indices)
    indices_to_update['context'].append(context_indices)
    indices_to_update['cls'].append(cls_indices)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('in_directory', help='path containing input files, '
                        'assuming one file per MWE, one sentence per line')
    parser.add_argument('out_directory', help='path to write output files')
    parser.add_argument('--min', help='lowest BERT layer (default: 1)',
                        default=1, type=int)
    parser.add_argument('--max', help='highest BERT layer (default: 12)',
                        default=12, type=int)
    parser.add_argument('--combo', help="type of layer combinations: 'win' "
                        "(default) makes contiguous windows, 'full' makes "
                        "all unique combinations, 'debug' makes only the "
                        "longest combination", default='win',
                        choices=['win', 'full', 'debug'])
    parser.add_argument('--pooling', help="token-wise, layer-wise, and "
                        "sequence-wise pooling method(s), overriding "
                        "individual choices if set, defaults to None",
                        nargs='+', default=None,
                        choices=['avg', 'sum', 'max', 'concat'])
    parser.add_argument('--token', help="token-wise pooling method(s),"
                        " defaults to 'avg'", nargs='+', default=['avg'],
                        choices=['avg', 'sum', 'max', 'concat'])
    parser.add_argument('--layer', help="layer-wise pooling method(s),"
                        " defaults to 'avg'", nargs='+', default=['avg'], 
                        choices=['avg', 'sum', 'max', 'concat'])
    parser.add_argument('--type', help="method to create "
                        "type-level embeddings, defaults to 'avg'", 
                        nargs='+', default=['avg'], 
                        choices=['avg', 'sum', 'max', 'concat'])
    parser.add_argument('--len', help='minimum sequence length(s), '
                        'defaults to {2, 5, 10}', default=[2, 5, 10],
                        nargs='+', type=int)
    parser.add_argument('--num', help='number(s) of modeled sequences, '
                        'defualts to {10, 100}', default=[10, 100],
                        nargs='+', type=int)
    parser.add_argument('--preproc', help="pass --no-preproc to avoid "
                        "preprocessing filters", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--shuf', help="reshuffle examples from input files",
                        default=False,
                        action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()

    out_directory = args.out_directory

    # Get layer combinations to be used for embeddings
    min_layer = args.min
    max_layer = args.max
    how_combos = args.combo
    layer_combos = get_layer_combos(min_layer=min_layer,
                                    max_layer=max_layer,
                                    how=how_combos)

    # Load MWE targets
    directory = args.in_directory
    targets = os.listdir(directory)
    targets = sorted([t[:-4].replace('_', ' ') for t in targets])

    # Check existing output and filter out targets
    ex_directory = args.out_directory
    ex_targets = os.listdir(ex_directory)
    ex_targets = [t[:-4] for t in ex_targets]
    skipping = [t for t in targets if t in ex_targets]
    targets = [t for t in targets if t not in ex_targets]

    # Define parameters investigated in the experiment
    master_poolings = args.pooling
    if master_poolings is not None:
        token_poolings = layer_poolings = typelevel_poolings = master_poolings
    else:
        token_poolings = args.token
        layer_poolings = args.layer
        typelevel_poolings = args.type        
    if args.preproc:
        prefilters = ['none', 'position', 'punct', 'all']
    else:
        prefilters = ['none']
    min_seq_lens = args.len  # minimum space-separated tokens in example
    n_seqs = args.num  # number of examples modeled
    max_n_seq = max(n_seqs)
    shuf = args.shuf

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.info(f'Loaded {len(targets)} targets from {directory}.')
    logging.info(f'Output directory: {out_directory}.')
    logging.info(f'Number of layer combinations: {len(layer_combos)}.')
    logging.info(f"Token pooling methods: {', '.join(token_poolings)}.")
    logging.info(f"Layer pooling methods: {', '.join(layer_poolings)}.")
    logging.info(f"Type-level methods: {', '.join(typelevel_poolings)}.")
    logging.info(f"Same pooling for token, layer, and type-level: "
                 f"{'no' if master_poolings is None else 'yes'}.")
    logging.info(f"Preprocessing filters: {', '.join(prefilters)}.")
    logging.info(f"Minimum sequence lengths: "
                 f"{', '.join([str(l) for l in min_seq_lens])}.")
    logging.info(f"Number of modeled sequences: "
                 f"{', '.join([str(n) for n in n_seqs])}.")
    logging.info(f"Shuffle sequences before processing: {shuf}.")
    
    for target in skipping:
        logging.info(f'Skipping "{target}".')

    for target in targets:

        # STEP 1. Read and preprocess examples
        logging.info(f'Processing "{target}".')
        examples = read_examples(target, directory)
        if shuf:
            np.random.shuffle(examples)

        # Apply preprocessing filter combos to loaded examples
        filtered_seqs = filter_examples(target=target,
                                        examples=examples,
                                        prefilters=prefilters,
                                        max_n_seq=max_n_seq,
                                        min_seq_lens=min_seq_lens)

        unique_seqs = get_unique_filtered_seqs(filtered_seqs)

        # STEP 2. Model all unique examples
        modif_ids, head_ids = get_target_ids(target)
        n_modif_tokens = len(modif_ids)
        n_head_tokens = len(head_ids)

        err_tokens = ['OOV:']
        if n_modif_tokens > 1:
            err_tokens.append(f'Modifier split into {n_modif_tokens} tokens.')
        if n_head_tokens > 1:
            err_tokens.append(f'Head split into {n_head_tokens} tokens.')
        if len(err_tokens) > 1:
            logging.warning(' '.join(err_tokens))

        indices_in_seq = {'modif': [], 'head': [], 'mwe': [], 'context': [], 
                          'cls': []}
        bert_tokens = []
        model_outs = []
        seq2model_out = {}

        for seq_idx in unique_seqs:
            seq = examples[seq_idx]
            seq_ids = tokenizer.encode(seq, return_tensors='pt')

            try: 
                model_out = model(seq_ids, output_hidden_states=True)
            except RuntimeError:
                # catches BERT modeling errors
                logging.warning(f'Skipping sequence {seq_idx} on modeling.')
                continue

            try:
                update_all_indices_in_seq(modif_ids=modif_ids,
                                          head_ids=head_ids,
                                          seq_ids=seq_ids,
                                          indices_to_update=indices_in_seq)
            except ValueError as e:
                print(e)
                # catches embedding extraction errors, eg special characters
                # messing up tokenization
                logging.warning(f'Skipping sequence {seq_idx} after modeling.')
                continue
            # the next 3 lines used to come before update_all_indices_in_seq    
            bert_tokens.append(seq_ids)
            model_outs.append(model_out)
            seq2model_out[seq_idx] = len(model_outs) - 1

        logging.info(f'Modeled {len(model_outs)} sequences.')

        # STEP 3. Generate embeddings and measure distances
        master_dists = []

        problems_to_log = []  # store potentially repetitive problems

        for layer_combo in layer_combos:
            for layer_pooling in layer_poolings:
                if master_poolings is not None:
                    token_poolings = [layer_pooling]
                    typelevel_poolings = [layer_pooling]
                for token_pooling in token_poolings:

                    # Compute embeddings for a given layers/pooling setup
                    embs = defaultdict(list)

                    for emb_type in indices_in_seq: # modif head mwe context cls
                        for occurrences, model_out in zip(indices_in_seq[emb_type],
                                                          model_outs):
                            embs[emb_type].append([])
                            for occurrence in occurrences:
                                emb = get_emb(model_out, 
                                              token_indices=occurrence, 
                                              layer_indices=layer_combo, 
                                              layer_pooling=layer_pooling,
                                              token_pooling=token_pooling)
                                embs[emb_type][-1].append(emb)                

                    # Compute and store distances. The embeddings computed above
                    # are used for all variations in parameters
                    dist_combos = list(combinations(embs.keys(), 2))

                    for prefilter in prefilters:
                        for min_seq_len in min_seq_lens:
                            for n_seq in n_seqs:

                                seq_indices = filtered_seqs[prefilter][min_seq_len][:n_seq]
                                # remove sequences skipped due to errors
                                seq_indices = [seq_idx for seq_idx in seq_indices
                                               if seq_idx in seq2model_out]

                                if len(seq_indices) < n_seq:
                                    problems_to_log.append(f'Limited sequences: '
                                        f'{len(seq_indices)}, expecting {n_seq} '
                                        f'for {prefilter}/{min_seq_len}.')
                                    
                                layer_combo_str = f'{min(layer_combo)}-'\
                                                  f'{max(layer_combo)}'
                                dists_meta = {'layer_combo': layer_combo_str,
                                              'layer_pooling': layer_pooling,
                                              'token_pooling': token_pooling,
                                              'prefilter': prefilter,
                                              'min_seq_len': min_seq_len,
                                              'n_seqs': n_seq}

                                for emb_type1, emb_type2 in dist_combos:
                                    dists_meta['emb1'] = emb_type1
                                    dists_meta['emb2'] = emb_type2

                                    dists = dists_meta.copy()
                                    token_dists = []
                                    all_emb1 = []
                                    all_emb2 = []

                                    for seq_idx in seq_indices:
                                        emb1 = embs[emb_type1][seq2model_out[seq_idx]]
                                        emb2 = embs[emb_type2][seq2model_out[seq_idx]]

                                        # accounts for multiple occurrences per line
                                        for i in range(len(emb1)):

                                            # compute token-level cosines
                                            cos = cosine(emb1[i], emb2[i])
                                            token_dists.append(cos)

                                            # prepare for type-level distances
                                            all_emb1.append(emb1[i])
                                            all_emb2.append(emb2[i])    

                                    # compute final token-level distances
                                    dists['type'] = 'token'
                                    dists['value'] = np.mean(token_dists)
                                    master_dists.append(dists)

                                    # compute type-level distances
                                    for typelevel_pooling in typelevel_poolings:
                                        dists = dists_meta.copy()
                                        dists['type'] = f'type-{typelevel_pooling}'
                                        type_emb1 = apply_pooling(all_emb1, pooling=typelevel_pooling)
                                        type_emb2 = apply_pooling(all_emb2, pooling=typelevel_pooling)
                                        dists['value'] = cosine(type_emb1, type_emb2)
                                        master_dists.append(dists)

        df = pd.DataFrame(master_dists)
        df.to_pickle(os.path.join(out_directory, f'{target}.pkl'))
        
        problems_to_log = sorted(set(problems_to_log))
        for problem in problems_to_log:
            logging.warning(problem)
        logging.info('Output written.')

    
if __name__ == '__main__':
    main()
