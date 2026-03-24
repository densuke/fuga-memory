"""config.py のユニットテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fuga_memory.config import Config, config_file_paths


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


class TestConfigFilePaths:
    def test_macos_appdata_is_first_on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        paths = config_file_paths()
        assert "Library/Application Support" in str(paths[0])

    def test_non_darwin_does_not_include_appdata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        paths = config_file_paths()
        assert not any("Library/Application Support" in str(p) for p in paths)

    def test_xdg_config_home_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/custom/config")
        paths = config_file_paths()
        assert any("/custom/config" in str(p) for p in paths)

    def test_default_xdg_path_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        paths = config_file_paths()
        assert any(".config/fuga-memory" in str(p) for p in paths)

    def test_home_dotfile_is_always_last(self) -> None:
        paths = config_file_paths()
        assert paths[-1] == Path.home() / ".fuga-memory.toml"

    def test_returns_at_least_two_paths(self) -> None:
        paths = config_file_paths()
        assert len(paths) >= 2


class TestConfigLoadFromToml:
    def test_load_from_explicit_path(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nrrf_k = 99\n")
        config = Config.load(config_path=toml_file)
        assert config.rrf_k == 99

    def test_load_flat_keys_without_section(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("rrf_k = 77\n")
        config = Config.load(config_path=toml_file)
        assert config.rrf_k == 77

    def test_db_path_tilde_is_expanded(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text('[fuga-memory]\ndb_path = "~/memories.db"\n')
        config = Config.load(config_path=toml_file)
        assert config.db_path == Path.home() / "memories.db"

    def test_all_fields_loadable_from_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            "[fuga-memory]\n"
            'model_name = "some/model"\n'
            "thread_workers = 8\n"
            "rrf_k = 42\n"
            "decay_halflife_days = 14\n"
            "default_top_k = 10\n"
        )
        config = Config.load(config_path=toml_file)
        assert config.model_name == "some/model"
        assert config.thread_workers == 8
        assert config.rrf_k == 42
        assert config.decay_halflife_days == 14
        assert config.default_top_k == 10

    def test_missing_explicit_config_file_uses_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.toml"
        config = Config.load(config_path=missing)
        assert config.rrf_k == 60

    def test_partial_toml_keeps_unspecified_defaults(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nrrf_k = 42\n")
        config = Config.load(config_path=toml_file)
        assert config.decay_halflife_days == 30
        assert config.default_top_k == 5

    def test_section_takes_precedence_over_flat_keys(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("rrf_k = 10\n[fuga-memory]\nrrf_k = 20\n")
        config = Config.load(config_path=toml_file)
        assert config.rrf_k == 20

    def test_toml_syntax_error_includes_path(self, tmp_path: Path) -> None:
        """TOML 構文エラー時にファイルパスを含むエラーになる。"""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("invalid toml [[[")
        with pytest.raises(ValueError, match=str(bad_toml)):
            Config.load(config_path=bad_toml)

    def test_toml_thread_workers_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nthread_workers = 0\n")
        with pytest.raises(ValueError, match="thread_workers"):
            Config.load(config_path=toml_file)

    def test_toml_rrf_k_negative_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nrrf_k = -1\n")
        with pytest.raises(ValueError, match="rrf_k"):
            Config.load(config_path=toml_file)

    def test_toml_decay_halflife_zero_raises(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\ndecay_halflife_days = 0\n")
        with pytest.raises(ValueError, match="decay_halflife_days"):
            Config.load(config_path=toml_file)

    def test_toml_rrf_k_zero_is_valid(self, tmp_path: Path) -> None:
        """rrf_k = 0 は許容される（最小値は 0）。"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nrrf_k = 0\n")
        config = Config.load(config_path=toml_file)
        assert config.rrf_k == 0


class TestConfigEnvOverride:
    """環境変数による設定上書き（全フィールド対象）。"""

    def _no_file(self, tmp_path: Path) -> Path:
        """存在しないパスを返す（ファイル読み込みをスキップさせる）。"""
        return tmp_path / "nonexistent.toml"

    def test_db_path_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_db = str(tmp_path / "env.db")
        monkeypatch.setenv("FUGA_MEMORY_DB_PATH", custom_db)
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.db_path == Path(custom_db)

    def test_model_name_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_MODEL_NAME", "custom/model")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.model_name == "custom/model"

    def test_thread_workers_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_THREAD_WORKERS", "8")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.thread_workers == 8

    def test_rrf_k_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_RRF_K", "80")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.rrf_k == 80

    def test_decay_halflife_days_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DECAY_HALFLIFE_DAYS", "14")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.decay_halflife_days == 14

    def test_default_top_k_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEFAULT_TOP_K", "10")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.default_top_k == 10

    def test_env_overrides_toml_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[fuga-memory]\nrrf_k = 99\n")
        monkeypatch.setenv("FUGA_MEMORY_RRF_K", "42")
        config = Config.load(config_path=toml_file)
        assert config.rrf_k == 42

    def test_invalid_thread_workers_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FUGA_MEMORY_THREAD_WORKERS", "notanint")
        with pytest.raises(ValueError, match="FUGA_MEMORY_THREAD_WORKERS"):
            Config.load(config_path=self._no_file(tmp_path))

    def test_invalid_rrf_k_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_RRF_K", "notanint")
        with pytest.raises(ValueError, match="FUGA_MEMORY_RRF_K"):
            Config.load(config_path=self._no_file(tmp_path))

    def test_thread_workers_zero_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FUGA_MEMORY_THREAD_WORKERS=0 は範囲エラー。"""
        monkeypatch.setenv("FUGA_MEMORY_THREAD_WORKERS", "0")
        with pytest.raises(ValueError, match="FUGA_MEMORY_THREAD_WORKERS"):
            Config.load(config_path=self._no_file(tmp_path))

    def test_rrf_k_negative_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FUGA_MEMORY_RRF_K=-1 は範囲エラー（ゼロ除算防止）。"""
        monkeypatch.setenv("FUGA_MEMORY_RRF_K", "-1")
        with pytest.raises(ValueError, match="FUGA_MEMORY_RRF_K"):
            Config.load(config_path=self._no_file(tmp_path))

    def test_rrf_k_zero_is_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FUGA_MEMORY_RRF_K=0 は許容される。"""
        monkeypatch.setenv("FUGA_MEMORY_RRF_K", "0")
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.rrf_k == 0

    def test_decay_halflife_zero_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FUGA_MEMORY_DECAY_HALFLIFE_DAYS=0 は範囲エラー。"""
        monkeypatch.setenv("FUGA_MEMORY_DECAY_HALFLIFE_DAYS", "0")
        with pytest.raises(ValueError, match="FUGA_MEMORY_DECAY_HALFLIFE_DAYS"):
            Config.load(config_path=self._no_file(tmp_path))

    def test_defaults_when_env_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in (
            "FUGA_MEMORY_DB_PATH",
            "FUGA_MEMORY_MODEL_NAME",
            "FUGA_MEMORY_THREAD_WORKERS",
            "FUGA_MEMORY_RRF_K",
            "FUGA_MEMORY_DECAY_HALFLIFE_DAYS",
            "FUGA_MEMORY_DEFAULT_TOP_K",
        ):
            monkeypatch.delenv(key, raising=False)
        config = Config.load(config_path=self._no_file(tmp_path))
        assert config.model_name == "cl-nagoya/ruri-v3-310m"
        assert config.rrf_k == 60


class TestConfigFromEnv:
    """from_env() が環境変数のみを適用し、設定ファイルを読まないことを確認。"""

    def test_from_env_returns_config(self) -> None:
        config = Config.from_env()
        assert isinstance(config, Config)

    def test_from_env_applies_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_MODEL_NAME", "compat/model")
        config = Config.from_env()
        assert config.model_name == "compat/model"

    def test_from_env_does_not_read_config_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() は設定ファイルを読まない（環境変数のみ）。"""
        # XDG_CONFIG_HOME を tmp_path に向けて設定ファイルを置いても無視されること
        config_dir = tmp_path / "fuga-memory"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text("[fuga-memory]\nrrf_k = 999\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.delenv("FUGA_MEMORY_RRF_K", raising=False)

        config = Config.from_env()
        assert config.rrf_k == 60  # デフォルト値のまま（ファイルは読まれない）
