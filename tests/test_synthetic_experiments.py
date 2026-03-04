import torch

from medrap.experiments.synthetic import (
    ContinuousTargetRecipe,
    CorruptedKeyRetriever,
    LabelCollisionRecipe,
    LearnedKeyQueryRetriever,
    OracleRetriever,
    build_toy_corpus,
    mean_absolute_error,
    oracle_recall_at_1,
    retrieval_is_sample_dependent,
)


def test_oracle_retriever_is_sample_dependent_and_perfect_recall() -> None:
    target_doc_ids = torch.LongTensor([0, 1, 2])
    result = OracleRetriever(top_k=1).retrieve(target_doc_ids=target_doc_ids)

    assert retrieval_is_sample_dependent(result)
    assert float(oracle_recall_at_1(result, target_doc_ids)) == 1.0


def test_label_collision_recipe_reproduces_equivalence_class_failure_mode() -> None:
    recipe = LabelCollisionRecipe(positive_drug_ids={0, 2})

    assert recipe.target(0) == 1
    assert recipe.target(2) == 1
    assert recipe.target(1) == 0
    assert 0 != 2  # distinct drugs collapse to same supervision target


def test_continuous_target_recipe_disambiguates_colliding_binary_labels() -> None:
    recipe = ContinuousTargetRecipe(values_by_drug_id={0: 0.1, 2: 0.8})

    t0 = recipe.target(0)
    t2 = recipe.target(2)
    assert t0 != t2

    preds = torch.tensor([t0, t2])
    refs = torch.tensor([0.1, 0.8])
    assert float(mean_absolute_error(preds, refs)) == 0.0


def test_learned_query_key_retriever_can_learn_alignment() -> None:
    corpus = build_toy_corpus(num_docs=4, dim=4, seed=7)
    retriever = LearnedKeyQueryRetriever(corpus=corpus, query_dim=4, top_k=1)

    queries = torch.eye(4)
    target_doc_ids = torch.LongTensor([0, 1, 2, 3])

    with torch.no_grad():
        initial_scores = retriever.score(queries)
        initial_recall = (initial_scores.argmax(dim=-1) == target_doc_ids).float().mean()

    optimizer = torch.optim.SGD(retriever.parameters(), lr=0.5)
    for _ in range(150):
        optimizer.zero_grad()
        scores = retriever.score(queries)
        loss = torch.nn.functional.cross_entropy(scores, target_doc_ids)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        trained_scores = retriever.score(queries)
        trained_recall = (trained_scores.argmax(dim=-1) == target_doc_ids).float().mean()

    assert float(trained_recall) >= float(initial_recall)
    assert float(trained_recall) == 1.0


def test_corrupted_key_retriever_degrades_alignment() -> None:
    corpus = build_toy_corpus(num_docs=4, dim=4, seed=1)
    eye = torch.eye(4)
    for idx, doc in enumerate(corpus.documents):
        doc.key = eye[idx].clone()

    base = LearnedKeyQueryRetriever(corpus=corpus, query_dim=4, top_k=1)

    with torch.no_grad():
        # Make query projection identity so retrieval quality depends purely on key geometry.
        base.query_proj.weight.copy_(torch.eye(4))

    queries = eye.clone()
    target_doc_ids = torch.arange(4)

    base_result = base.retrieve(queries)
    base_recall = oracle_recall_at_1(base_result, target_doc_ids)

    g = torch.Generator().manual_seed(13)
    random_rot = torch.randn((4, 4), generator=g)
    corrupted = CorruptedKeyRetriever(base=base, corruption=random_rot)
    corrupted_result = corrupted.retrieve(queries)
    corrupted_recall = oracle_recall_at_1(corrupted_result, target_doc_ids)

    assert float(base_recall) == 1.0
    assert float(corrupted_recall) < float(base_recall)
