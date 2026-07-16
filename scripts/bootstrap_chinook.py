"""Download the Chinook sample SQL script and hydrate a local SQLite file.

Usage:
    python scripts/bootstrap_chinook.py             # creates ./chinook.db
    python scripts/bootstrap_chinook.py --force     # overwrite existing file
    python scripts/bootstrap_chinook.py --db data/chinook.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.request
from pathlib import Path

CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)


def bootstrap(db_path: Path, force: bool = False) -> Path:
    """Download Chinook_Sqlite.sql and execute it into ``db_path``."""
    if db_path.exists() and not force:
        print(
            f"{db_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return db_path
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {CHINOOK_SQL_URL} ...")
    with urllib.request.urlopen(CHINOOK_SQL_URL) as response:  # noqa: S310
        sql_script = response.read().decode("utf-8")

    print(f"Creating {db_path} ...")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(sql_script)
        conn.commit()
    finally:
        conn.close()

    # Sanity check.
    conn = sqlite3.connect(db_path)
    try:
        (n_tracks,) = conn.execute("SELECT COUNT(*) FROM Track").fetchone()
        (n_customers,) = conn.execute(
            "SELECT COUNT(*) FROM Customer"
        ).fetchone()
    finally:
        conn.close()
    print(f"Loaded {n_tracks} tracks and {n_customers} customers.")
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="chinook.db",
        help="Path to the output SQLite file (default: ./chinook.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the database file if it already exists",
    )
    args = parser.parse_args()
    bootstrap(Path(args.db), force=args.force)


if __name__ == "__main__":
    main()
