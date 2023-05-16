# Compositionality
Repo for compositionality scale project

See NOTES here: https://docs.google.com/document/d/1HdaP62N9srEGwFOJUDZxRy9jVBl0sFvz_BaxQGi-uRY/edit
# 20230511
To do:\
Play around with different LMs.
Including GPT3 - zero-shot classification of compositionality. (Can be used as a baseline)\
Read papers on human processing of compositionality scale: NC => PC => C or PC => NC => C?\
Readingi in the wider literature regarding topics surronding promting, noun compounds, compositionality

Construct dataset(s) of cosine similarities between words in the noun compounds - to compare with cosine similarities between words in their synonyms

# 20230516
Vector Norms

Current literature shows that BERT knows compositionality of compound semantics. 

(Miletic and Schulte im Walde, 2023) found that first few layers knows but then loses knowledge. 

Our research scope:
* which layer stores compositionality information best
* check if compositionality information is in the norm (magnitude) since cosine similarity ignores the information stored in the vector norm.
We are measuring difference in distance of vectors

Results potentially could verify findings of Klubicka, Kelleher (2022) if nothing else new is found

## Datatset
Possible options:
* NCTTI
* COMPS
* Compositionality of Nominal Compounds - Datasets https://pageperso.lis-lab.fr/carlos.ramisch/?page=downloads/compounds

ASK ALINE ABOUT HUMAN EVALUATIONS/ORIGINAL ANNOTATOR RESULTS - WHAT EACH ANNOTATOR ACTUALLY SAID (nice to have in addition to average annotator performance).


