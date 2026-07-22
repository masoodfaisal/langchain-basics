#!/usr/bin/env bash

set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-$project_dir/.venv}"
db_path="${CHINOOK_DB_PATH:-$project_dir/chinook.db}"
requirements_file="$project_dir/requirements.txt"
check_only=false
force_db=false

if [[ "$venv_dir" != /* ]]; then
  venv_dir="$project_dir/$venv_dir"
fi
if [[ "$db_path" != /* ]]; then
  db_path="$project_dir/$db_path"
fi

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--check-only] [--force-db]

Create and validate the local development environment.

Options:
  --check-only  Validate without creating files, installing packages, or
                downloading the Chinook database.
  --force-db    Re-download and replace the Chinook database.
  -h, --help    Show this help message.

Environment variables:
  VENV_DIR         Virtual environment location (default: .venv)
  CHINOOK_DB_PATH  SQLite database location (default: chinook.db)
EOF
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

log() {
  printf '==> %s\n' "$*"
}

while (($# > 0)); do
  case "$1" in
    --check-only)
      check_only=true
      ;;
    --force-db)
      force_db=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown option: $1"
      ;;
  esac
  shift
done

if $check_only && $force_db; then
  fail "--check-only and --force-db cannot be used together"
fi

python_bin="python3.13"
"$python_bin" --version >/dev/null 2>&1 || fail \
  "cannot run python3.13; install Python 3.13 and try again"

"$python_bin" - <<'PY' || fail "install Python 3.13 and try again"
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit(
        f"found Python {sys.version_info.major}.{sys.version_info.minor}, expected 3.13"
    )
PY

cd "$project_dir"

if [[ ! -x "$venv_dir/bin/python" ]]; then
  if $check_only; then
    fail "virtual environment not found at $venv_dir"
  fi
  log "Creating Python 3.13 virtual environment at $venv_dir"
  "$python_bin" -m venv "$venv_dir"
else
  log "Using existing virtual environment at $venv_dir"
fi

venv_python="$venv_dir/bin/python"

"$venv_python" - "$venv_dir" <<'PY' || fail "the virtual environment is invalid; remove it and run setup again"
import sys
from pathlib import Path

expected_prefix = Path(sys.argv[1]).resolve()
actual_prefix = Path(sys.prefix).resolve()
if actual_prefix != expected_prefix:
    raise SystemExit(f"venv prefix is {actual_prefix}, expected {expected_prefix}")
if sys.version_info[:2] != (3, 13):
    raise SystemExit(
        f"venv uses Python {sys.version_info.major}.{sys.version_info.minor}, expected 3.13"
    )
PY

if ! $check_only; then
  command -v sfw >/dev/null 2>&1 || fail \
    "sfw is required for protected dependency installation: https://github.com/SocketDev/sfw-free"
  log "Installing Python dependencies from requirements.txt"
  sfw "$venv_python" -m pip install --requirement "$requirements_file"
fi

log "Validating installed Python dependencies"
"$venv_python" -m pip check
"$venv_python" - "$requirements_file" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from packaging.requirements import Requirement

requirements_file = Path(sys.argv[1])
errors: list[str] = []

for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        continue
    requirement = Requirement(line)
    if requirement.marker and not requirement.marker.evaluate():
        continue
    try:
        installed_version = version(requirement.name)
    except PackageNotFoundError:
        errors.append(f"{requirement.name} is not installed")
        continue
    if requirement.specifier and installed_version not in requirement.specifier:
        errors.append(
            f"{requirement.name} {installed_version} does not satisfy "
            f"{requirement.specifier}"
        )

if errors:
    raise SystemExit("\n".join(errors))

print(f"Validated {requirements_file.name}.")
PY

if ! $check_only && { $force_db || [[ ! -f "$db_path" ]]; }; then
  log "Installing the Chinook SQLite database"
  bootstrap_args=(--db "$db_path")
  if $force_db; then
    bootstrap_args+=(--force)
  fi
  "$venv_python" scripts/bootstrap_chinook.py "${bootstrap_args[@]}"
elif ! $check_only; then
  log "Using existing Chinook database at $db_path"
fi

log "Validating the Chinook SQLite database"
"$venv_python" - "$db_path" <<'PY'
from pathlib import Path
import sqlite3
import sys

db_path = Path(sys.argv[1])
if not db_path.is_file():
    raise SystemExit(f"Chinook database not found at {db_path}")

required_tables = {
    "Album",
    "Artist",
    "Customer",
    "Employee",
    "Genre",
    "Invoice",
    "InvoiceLine",
    "MediaType",
    "Playlist",
    "PlaylistTrack",
    "Track",
}

with sqlite3.connect(db_path) as connection:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing_tables = sorted(required_tables - tables)
    if missing_tables:
        raise SystemExit(
            f"Chinook database is missing tables: {', '.join(missing_tables)}"
        )
    track_count = connection.execute("SELECT COUNT(*) FROM Track").fetchone()[0]
    customer_count = connection.execute("SELECT COUNT(*) FROM Customer").fetchone()[0]

if integrity != "ok":
    raise SystemExit(f"SQLite integrity check failed: {integrity}")
if track_count != 3503 or customer_count != 59:
    raise SystemExit(
        "unexpected Chinook row counts: "
        f"tracks={track_count} (expected 3503), "
        f"customers={customer_count} (expected 59)"
    )

print(f"Validated {db_path} ({track_count} tracks, {customer_count} customers).")
PY

log "Setup complete"
printf 'Activate the environment with: source %q\n' "$venv_dir/bin/activate"
