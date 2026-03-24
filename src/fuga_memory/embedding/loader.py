"""バックグラウンドでモデルをロードする ModelLoader。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from fuga_memory.embedding.encoder import Encoder, RuriEncoder
from fuga_memory.exceptions import ModelLoadError


class ModelLoader:
    """埋め込みモデルを ThreadPoolExecutor でバックグラウンドロードし、キャッシュするクラス。

    初回 get_encoder() 呼び出し時にモデルをバックグラウンドでロードし、
    以降の呼び出しではキャッシュされたエンコーダを返す。
    """

    def __init__(self, model_name: str, thread_workers: int = 1) -> None:
        self._model_name = model_name
        self._thread_workers = thread_workers
        self._encoder: Encoder | None = None
        self._future: Future[Encoder] | None = None

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

        初回呼び出し時は ThreadPoolExecutor でバックグラウンドロードを実行し、
        結果が得られるまでブロックする。2回目以降はキャッシュを返す。

        Returns:
            Encoder プロトコルを満たすエンコーダオブジェクト。

        Raises:
            ModelLoadError: モデルのロードに失敗した場合。
        """
        if self._encoder is not None:
            return self._encoder

        try:
            with ThreadPoolExecutor(max_workers=self._thread_workers) as executor:
                future: Future[Encoder] = executor.submit(RuriEncoder, self._model_name)
                encoder = future.result()
        except Exception as exc:
            raise ModelLoadError(
                f"モデル '{self._model_name}' のロードに失敗しました: {exc}"
            ) from exc

        self._encoder = encoder
        return self._encoder
