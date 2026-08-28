"""One TLS context that works under every interpreter on this machine.

The conda environments here carry an OpenSSL whose compiled-in CA path points at a stale
location (``/mnt/zhang-nas/...``), so ``urlopen`` fails verification with
``CERTIFICATE_VERIFY_FAILED`` even though the machine has a perfectly good CA bundle. The
system python has correct paths and the conda ones do not, which makes the failure look
like a network problem rather than a configuration one.

Resolved once, here, rather than by setting ``SSL_CERT_FILE`` in every shell that happens
to invoke the pipeline.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Optional

# Checked in order; the first readable bundle wins.
_SYSTEM_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
)

_context: Optional[ssl.SSLContext] = None


def ca_bundle() -> Optional[str]:
    """Path to a usable CA bundle, or ``None`` to accept OpenSSL's own defaults."""
    override = os.environ.get("SSL_CERT_FILE")
    if override and Path(override).is_file():
        return override

    try:
        import certifi

        bundle = certifi.where()
        if Path(bundle).is_file():
            return bundle
    except ImportError:
        pass

    for bundle in _SYSTEM_BUNDLES:
        if Path(bundle).is_file():
            return bundle

    # Only trust the built-in paths if they actually resolve to something.
    paths = ssl.get_default_verify_paths()
    if paths.cafile and Path(paths.cafile).is_file():
        return paths.cafile
    return None


def context() -> ssl.SSLContext:
    """A verifying TLS context, cached. Verification is never disabled."""
    global _context
    if _context is None:
        bundle = ca_bundle()
        _context = ssl.create_default_context(cafile=bundle)
    return _context
