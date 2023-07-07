import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from itertools import combinations

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_target_ids(target):
    """Retrieves modifier and head IDs as tokenized by BERT.
    
    Args:
        target: Target MWE, assuming the format "<modifier> SPACE <head>".
    
    Returns:
        A tuple of 2 lists, correpsonding to modifier and head IDs provided by
        BERT's tokenizer. The lists contain a single ID if the whole token is 
        known to BERT; they contain multiple IDs if the original token is 
        split up by BERT.
    """
    
    modif, head = target.split()
    modif_ids = tokenizer.encode(modif)[1:-1]
    head_ids = tokenizer.encode(head)[1:-1]
    
    return modif_ids, head_ids


def get_indices_in_seq(modif_ids, head_ids, seq_ids):
    """Retrieves head and modifier token indices in a BERT-tokenized sequence.
    
    Args:
        modif_ids: List of modifier token IDs.
        head_ids: List of head token IDs.
        seq_ids: List of token IDs for the full tokenized sequence.
            
    Returns:
        A tuple of 2 lists, corresponding to modifier and head token indices.
        
    Raises:
        ValueError: If MWE token IDs are not found in the tokenized sequence.
    """
    
    if isinstance(seq_ids, torch.Tensor):
        seq_ids = seq_ids.flatten().tolist()
    
    # The target span of token IDs (modifier + head) is compared against 
    # all contiguous spans of token IDs of the same length in the tokenized
    # sequence (sliding window over tokens).
    target_ids = modif_ids + head_ids
    len_t = len(target_ids)
    len_s = len(seq_ids)
    spans = [seq_ids[i:i+len_t] for i in range(len_s-len_t)]
    
    target_indices = [i for i, span in enumerate(spans) if span == target_ids]
    if len(target_indices) < 1:
        raise ValueError('MWE tokens not found in tokenized sequence.')
        
    # The index of a target span is also the index of the first token in it.
    # This is used to reconstruct the indices of all modifier/head tokens.
    modif_indices = []
    head_indices = []
    len_m = len(modif_ids)
    for idx in target_indices:
        modif = list(range(idx, idx+len_m))
        head = list(range(idx+len_m, idx+len_t))
        modif_indices.append(modif)
        head_indices.append(head)
    
    assert len(modif_indices) == len(head_indices)
    
    return modif_indices, head_indices


def apply_pooling(vecs, pooling='avg'):
    """Computes a pooled representation from multiple input vectors.
    
    Args:
        vecs: A sequence of tensors to be pooled.
        pooling: Pooling method, one of ['avg', 'sum', 'max', 'concat', None].
            Defaults to 'avg'. None returns the first tensor in input sequence.
    
    Returns:
        A tensor of shape (hidden_size). If pooling method is 'concat', 
        the shape is (n_vectors * hidden_size).
    """
    
    if pooling == 'avg':
        vec = torch.stack(vecs).mean(dim=0)
    elif pooling == 'sum':
        vec = torch.stack(vecs).sum(dim=0)
    elif pooling == 'max':
        vec = torch.stack(vecs).max(dim=0).values
    elif pooling == 'concat':
        vec = torch.cat(vecs)
    elif pooling is None:
        vec = vecs[0]
    else:
        raise ValueError("Pooling method must be one of "
                         "['avg', 'sum', 'max', 'concat', None].")
    
    return vec


def get_token_hs(model_out, token_idx, layer_idx=-1):
    """Retrieves the output of a single BERT layer for a BERT-tokenized token.
    
    Args:
        See get_emb() docstring.
    
    Returns:
        A tensor of shape (hidden_size).
    """
    if not isinstance(token_idx, int):
        raise TypeError('Token index must be an integer.')
        
    token_hs = model_out[2][layer_idx][0][token_idx]
    return token_hs


def get_hs(model_out, token_indices, layer_idx=-1, pooling='avg'):
    """Retrieves the output of a single BERT layer for a target in a sequence.
    
    The target (constituent, MWE etc.) can be comprised of one or more BERT
    tokens. If there are multiple tokens, token-wisen pooling is applied.
    
    Args:
        See get_emb() docstring.
        
    Returns:
        A tensor of shape (hidden_size). If pooling method is 'concat',
        the shape is (n_vectors * hidden_size).
    """
    
    token_hidden_states = []
    for token_idx in token_indices:
        token_hs = get_token_hs(model_out, token_idx, layer_idx)
        token_hidden_states.append(token_hs)
    
    if len(token_hidden_states) == 1:
        hs = token_hidden_states[0]
    else:
        hs = apply_pooling(token_hidden_states, pooling=pooling)
        
    return hs


def get_emb(model_out, token_indices, layer_indices=[-1], 
            token_pooling='avg', layer_pooling='avg'):
    """Computes an embedding for a target element from a BERT-encoded sequence.
    
    Args:
        model_out: A tensor containing the Huggingface Transformers output
            of the full sequence.
        token_indices: A list of indices of the target tokens in the 
            BERT-tokenized sequence.
        layer_indices: A list of indices of the hidden states based on which
            the embedding is computed. Possible values to include are in
            range [0, 12], where 0 is the initial embedding output, and 12 
            is the output of the last layer (default).
        token_pooling: Pooling method applied across BERT tokens comprising
            the target element, within the same layer. One of ['avg', 'sum',
            'max', 'concat', None]. Defaults to 'avg'. None returns the first 
            tensor in input sequence.
        layer_pooling: Pooling method applied across BERT layers for a pooled
            representation of the target element. Same methods as above.
            
    Returns:
        A tensor of shape (hidden_size). If 'concat' is applied as a pooling
        method, the shape is (n_vectors * hidden_size).
    
    """
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    elif not isinstance(layer_indices, list):
        raise TypeError("Layer indices must be a list or an integer.")
    
    hidden_states = []
    for layer_idx in layer_indices:
        hs = get_hs(model_out, token_indices, layer_idx, token_pooling)
        hidden_states.append(hs)
    
    if len(layer_indices) > 1:
        emb = apply_pooling(hidden_states, layer_pooling)
    else:
        emb = hidden_states[0]
        
    return emb
