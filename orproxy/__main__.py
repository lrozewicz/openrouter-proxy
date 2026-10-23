import logging
import os
import sys

import uvicorn

from .config import ConfigError, load_config
from .server import create_app


def main() -> None:
    path = os.environ.get("ORPROXY_CONFIG", sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    try:
        cfg = load_config(path)
    except (ConfigError, OSError, ValueError) as exc:
        source = "ORPROXY_CONFIG_YAML" if os.environ.get("ORPROXY_CONFIG_YAML", "").strip() else path
        print(f"config error ({source}): {exc}", file=sys.stderr)
        sys.exit(2)
    logging.basicConfig(level=cfg.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level=cfg.log_level.lower(),
                access_log=False, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
