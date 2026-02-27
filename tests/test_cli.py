from clinical_retrieval_pt.cli import eval_main, main, train_main


def test_medrap_train_cli_runs_with_overrides() -> None:
    assert main(["train", "run_smoke=false"]) == 0


def test_medrap_eval_cli_runs_with_overrides() -> None:
    assert main(["eval", "run_smoke=false"]) == 0


def test_train_entrypoint_runs_with_hydra_overrides() -> None:
    assert train_main(["run_smoke=false"]) == 0


def test_eval_entrypoint_runs_with_hydra_overrides() -> None:
    assert eval_main(["run_smoke=false"]) == 0
