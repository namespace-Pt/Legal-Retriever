# Data

```python
# training
{
  "query": str,
  # positive documents
  "pos": List[str],
  # negative documents
  "neg": List[str],
  # the indices of the positive documents w.r.t. the corpus
  "pos_indices": Optional[List[int]],
  # the indices of the negative documents w.r.t. the corpus
  "neg_indices": Optional[List[int]],
  # the scores of the positive documents, used in distillation
  "pos_scores": Optional[List[float]],
  # the scores of the negative documents, used in distillation
  "neg_scores": Optional[List[float]],
  # list of answers for the query
  "answers": Optional[List[str]],
}

# evaluation
{
  "query": str,
  # positive documents
  "pos": Optional[List[str]],
  # the indices of the positive documents w.r.t. the corpus, used in computing retrieval metrics like MRR and Recall
  "pos_indices": Optional[List[int]],
  # the relevance score of the positive documents (some documents may be more relevant), used in computing NDCG
  "pos_rels": Optional[List[float]],
  # list of answers for the query, used in computing QA-based metric
  "answers": Optional[List[str]],
}

# corpus
{
  "text": str
}
```