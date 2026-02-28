import importlib


def test_medrap_package_is_importable() -> None:
    module = importlib.import_module("medrap")
    assert module is not None
