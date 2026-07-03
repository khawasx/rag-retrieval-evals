from rag.evaluate import precision_at_k, recall_at_k, mrr


def test_retrieval_metrics():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d"}
    assert precision_at_k(retrieved, relevant, 5) == 0.4
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert mrr(retrieved, relevant) == 0.5
