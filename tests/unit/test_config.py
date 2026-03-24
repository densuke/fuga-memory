"""config.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from fuga_memory.config import Config


class TestConfigDefaults:
    def test_db_path_default_is_under_local_share(self) -> None:
        config = Config()
        assert ".local/share/fuga-memory" in str(config.db_path)

    def test_db_filename_is_memories_db(self) -> None:
        config = Config()
        assert config.db_path.name == "memories.db"

    def test_model_name_is_ruri_v3(self) -> None:
        config = Config()
        assert config.model_name == "cl-nagoya/ruri-v3-310m"

    def test_embedding_dim_is_768(self) -> None:
        config = Config()
        assert config.embedding_dim == 768

    def test_thread_workers_is_positive(self) -> None:
        config = Config()
        assert config.thread_workers >= 1

    def test_rrf_k_default(self) -> None:
        config = Config()
        assert config.rrf_k == 60

    def test_decay_halflife_days_default(self) -> None:
        config = Config()
        assert config.decay_halflife_days == 30

    def test_top_k_default(self) -> None:
        config = Config()
        assert config.default_top_k == 5


class TestConfigCustom:
    def test_custom_db_path(self, tmp_path: Path) -> None:
        custom_path = tmp_path / "custom.db"
        config = Config(db_path=custom_path)
        assert config.db_path == custom_path

    def test_custom_model_name(self) -> None:
        config = Config(model_name="some/other-model")
        assert config.model_name == "some/other-model"

    def test_custom_thread_workers(self) -> None:
        config = Config(thread_workers=4)
        assert config.thread_workers == 4


class TestConfigEnvOverride:
    def test_db_path_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_db = str(tmp_path / "env.db")
        monkeypatch.setenv("FUGA_MEMORY_DB_PATH", custom_db)
        config = Config.from_env()
        assert config.db_path == Path(custom_db)

    def test_model_name_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_MODEL_NAME", "custom/model")
        config = Config.from_env()
        assert config.model_name == "custom/model"

    def test_defaults_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FUGA_MEMORY_DB_PATH", raising=False)
        monkeypatch.delenv("FUGA_MEMORY_MODEL_NAME", raising=False)
        config = Config.from_env()
        assert config.model_name == "cl-nagoya/ruri-v3-310m"
