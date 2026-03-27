"""クロスプラットフォームプロセス起動ユーティリティ。

デーモンサーバーをデタッチされた子プロセスとして起動する。
親プロセスが終了しても子プロセスは継続して動作する。
"""

from __future__ import annotations

import subprocess
import sys


def spawn_daemon_process(port: int) -> None:
    """fuga-memory デーモンサーバーをデタッチ起動する。

    起動後、サーバーが応答可能になるまで待機しない。
    起動確認には client.py の ensure_daemon_running() を使うこと。

    Args:
        port: デーモンサーバーがリッスンするポート番号。

    Note:
        Windows対応は暫定実装・未検証。
        GitHub Actions の windows-latest VM での動作確認が必要。
    """
    cmd = [sys.executable, "-m", "fuga_memory.daemon.server", "--port", str(port)]

    if sys.platform == "win32":
        # NOTE: Windows暫定実装・未検証
        # DETACHED_PROCESS: コンソールウィンドウを持たない独立プロセスとして起動
        # CREATE_NEW_PROCESS_GROUP: 親の Ctrl+C シグナルを受け取らない
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        subprocess.Popen(
            cmd,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    else:
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
