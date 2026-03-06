from medrap.cli import eval_main, main, train_main


def test_medrap_train_cli_runs_with_overrides() -> None:
    assert main(["train"]) == 0


def test_medrap_eval_cli_runs_with_overrides() -> None:
    assert main(["eval"]) == 0


def test_train_entrypoint_runs_with_hydra_overrides() -> None:
    assert train_main([]) == 0


def test_eval_entrypoint_runs_with_hydra_overrides() -> None:
    assert eval_main([]) == 0
