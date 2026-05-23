"""Smoke tests — verify package imports and basic config loading."""

from __future__ import annotations


def test_package_imports() -> None:
    import soccerpredict

    assert isinstance(soccerpredict.__version__, str)
    assert soccerpredict.__version__.count(".") >= 1


def test_paths_resolve() -> None:
    from soccerpredict.utils import data_dir
    from soccerpredict.utils.paths import PROJECT_ROOT

    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()

    raw = data_dir("raw")
    assert raw.exists()
    assert raw.name == "raw"


def test_load_default_config() -> None:
    from soccerpredict.utils import load_config

    cfg = load_config()
    assert cfg.random_seed == 42
    assert len(cfg.competitions) >= 1
    assert cfg.default_competition.name
