"""fuga-memory デーモン公開 API。"""

from fuga_memory.daemon.client import ensure_daemon_running, send_save_request

__all__ = ["ensure_daemon_running", "send_save_request"]
