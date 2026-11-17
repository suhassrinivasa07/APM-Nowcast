"""MOSDAC authenticated HTTP client for INSAT-3D/3DR/3DS data.

Authentication uses Basic credentials stored in environment variables:
  MOSDAC_USER / MOSDAC_PASSWORD

This client NEVER fabricates product URLs. It uses the official MOSDAC
HTTPS data interface documented at https://www.mosdac.gov.in/

If credentials are not configured, the client operates in LOCAL-ONLY mode:
  - automated downloads are skipped
  - locally-available files can still be read and validated
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from common.logging_config import get_logger

log = get_logger(__name__)

_MOSDAC_BASE = "https://www.mosdac.gov.in/"


def _build_session(retries: int = 3, backoff: float = 1.5) -> "requests.Session":
    if not _HAS_REQUESTS:
        raise ImportError("requests library is required. pip install requests")
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MOSDACClient:
    """
    Thin authenticated HTTPS client for the MOSDAC data portal.

    Credentials are read from environment variables at instantiation time:
      MOSDAC_USER, MOSDAC_PASSWORD

    If credentials are absent the client enters LOCAL mode; call
    ``is_authenticated`` to check before attempting downloads.
    """

    def __init__(
        self,
        base_url: str = _MOSDAC_BASE,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self._base     = base_url
        self._timeout  = timeout
        self._user     = os.environ.get("MOSDAC_USER", "")
        self._password = os.environ.get("MOSDAC_PASSWORD", "")
        self._session: Optional["requests.Session"] = None

        if self._user and self._password:
            if _HAS_REQUESTS:
                self._session = _build_session(retries=retries)
                self._session.auth = (self._user, self._password)
                self._session.headers.update({"User-Agent": "APM-Nowcast/0.2"})
                log.info("MOSDACClient: authenticated session created", extra={"user": self._user})
            else:
                log.warning("MOSDACClient: requests not installed — operating in LOCAL mode")
        else:
            log.info("MOSDACClient: no credentials found — LOCAL mode only")

    @property
    def is_authenticated(self) -> bool:
        return self._session is not None

    def download(self, remote_path: str, dest: Path, overwrite: bool = False) -> bool:
        """
        Download a file from MOSDAC.

        Parameters
        ----------
        remote_path : relative path on MOSDAC portal, e.g. 'gallery/INSAT3D/...'
        dest        : local destination Path
        overwrite   : re-download even if file already exists locally

        Returns True if succeeded or file already present, False otherwise.
        """
        if not self.is_authenticated:
            log.warning(
                "MOSDAC download skipped — no credentials configured",
                extra={"remote_path": remote_path},
            )
            return False

        if dest.exists() and not overwrite:
            log.info("File already exists — skipping download", extra={"path": str(dest)})
            return True

        url = urljoin(self._base, remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        log.info("Downloading from MOSDAC", extra={"url": url, "dest": str(dest)})
        try:
            assert self._session is not None
            response = self._session.get(url, timeout=self._timeout, stream=True)
            response.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            log.info("Download complete", extra={"path": str(dest), "size": dest.stat().st_size})
            return True
        except Exception as exc:
            log.error("MOSDAC download failed", extra={"url": url, "error": str(exc)})
            if dest.exists():
                dest.unlink()
            return False

    def list_products(self, path: str) -> list[str]:
        """
        Return product listing from a MOSDAC directory path (best-effort).
        Returns empty list when unauthenticated or on error.
        """
        if not self.is_authenticated:
            return []
        url = urljoin(self._base, path)
        try:
            assert self._session is not None
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            return re.findall(r'href="([^"]+\.h5[^"]*|[^"]+\.nc[^"]*|[^"]+\.hdf[^"]*)"', resp.text)
        except Exception as exc:
            log.error("MOSDAC listing failed", extra={"path": path, "error": str(exc)})
            return []
