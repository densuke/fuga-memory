"""バックグラウンドでモデルをロードする ModelLoader。"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from fuga_memory.embedding.encoder import Encoder, RuriEncoder
from fuga_memory.exceptions import ModelLoadError


class ModelLoader:
    """埋め込みモデルを ThreadPoolExecutor でロードし、キャッシュするクラス。

    get_encoder() の初回呼び出し時に ThreadPoolExecutor でロードを開始し、
    結果が得られるまでブロックする。2回目以降はキャッシュを返す。

    複数スレッドから同時に呼び出された場合でも、モデルロードは1回だけ実行される。
    ロードに失敗した場合は次回呼び出しで再試行できる。
    """

    def __init__(self, model_name: str, thread_workers: int = 1) -> None:
        if thread_workers < 1:
            raise ValueError(f"thread_workers は 1 以上である必要があります: {thread_workers}")
        self._model_name = model_name
        self._thread_workers = thread_workers
        self._encoder: Encoder | None = None
        self._future: Future[Encoder] | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """使用するモデル名。"""
        return self._model_name

    @property
    def thread_workers(self) -> int:
        """ThreadPoolExecutor のワーカースレッド数。"""
        return self._thread_workers

    def get_encoder(self) -> Encoder:
        """ロード済みエンコーダを返す。

        初回呼び出し時は ThreadPoolExecutor でロードを実行し、
        結果が得られるまでブロックする。2回目以降はキャッシュを返す。
        複数スレッドから同時に呼び出されても、ロードは1度のみ実行される。
        ロードに失敗した場合は次回呼び出しで再試行できる。

        Returns:
            Encoder プロトコルを満たすエンコーダオブジェクト。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        # キャッシュ済みならロックなしで即返す（高速パス）
        if self._encoder is not None:
            return self._encoder

        with self._lock:
            # ロック取得後に再チェック（ダブルチェックロッキング）
            if self._encoder is not None:
                return self._encoder

            # 進行中の Future を共有して重複ロードを防ぐ
            if self._future is None:
                executor = ThreadPoolExecutor(max_workers=self._thread_workers)
                self._future = executor.submit(RuriEncoder, self._model_name)
                executor.shutdown(wait=False)

            future = self._future

        try:
            encoder = future.result()
        except Exception as exc:
            # 失敗した Future をリセットして次回呼び出しで再試行できるようにする
            with self._lock:
                if self._future is future:
                    self._future = None
            raise ModelLoadError(
                f"モデル '{self._model_name}' のロードに失敗しました: {exc}"
            ) from exc

        # ロード成功: エンコーダをキャッシュし、Future の参照を解放する
        with self._lock:
            if self._encoder is None:
                self._encoder = encoder
            self._future = None

        return self._encoder
