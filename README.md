# Compositionality
Repo for compositionality scale project

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
* which layer stores this information best
* check if this information in the norm (magnitude) since cosine similarity nullifies information stored in the vector norm.
