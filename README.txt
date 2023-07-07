The scripts contained in this directory enable the replication of the
experiments on compositionality prediction from pretrained BERT as presented
in the following paper:

    Miletic, F., and Schulte im Walde, S. (2023). A Systematic Search for
    Compound Semantics in Pretrained BERT Architectures. In Proc. EACL.

Author: Filip Miletic (filip.miletic@ims.uni-stuttgart.de)
Current version: 23 May 2023


### RUNNING THE PIPELINE ###

Use run_experiments.sh to run the core modeling and evaluation pipeline.
Note that the sample setup in the script corresponds to a subset of the 
experimental parameters presented in the paper.


### PROCESSING STEPS ###

The following scripts are executed (see the documentation of the individual
scripts for their arguments):

compos_bert.py
    Models compound occurrences, produces cosine-based compositionality
    estimates, and outputs them as Pandas dataframe pickles, one per compound.
    
compos_estimates.py
    Merges existing compound-level estimates, produces composite estimates
    (using basic composition functions), and outputs them in a single
    Pandas dataframe pickle.
    
eval.py
    Computes correlations between the gold standard ratings and the predicted
    values, and outputs them in a Pandas dataframe pickle.


### DATA REQUIREMENTS ###

The pipeline assumes the existence of:
- corpus examples for compounds
- gold standard compositionality ratings

Compound examples are expected as a set of text files, each of which
corresponds to a single compound and is named accordingly (flea_market.txt).
Each line in a file is fed into BERT as a single sequence to be modeled;
we used one corpus sentence per line.

The gold standard data is expected as a Pandas dataframe containing at least
the columns `target`, for target compounds; and `cpd`, `head`, and `modif`,
for compound-level, head, and modifier compositionality ratings. The set of
compounds is expected to be the same as those that were modeled.
The dataset used in the paper is provided as a dataframe (gold.pkl);
it includes the ratings from Reddy et al. (2011) and Cordeiro et al. (2019).
