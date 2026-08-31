"""Regression tests for the local Keycloak launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


START_KEYCLOAK = (
    Path(__file__).resolve().parents[1] / "keycloak-sso" / "start-keycloak.sh"
)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def test_non_running_container_uses_persisted_host_port_binding(
    tmp_path: Path,
) -> None:
    """A non-running container can lack runtime ports but remain startable."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_calls = tmp_path / "docker-calls.log"

    _write_executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >> "$DOCKER_CALL_LOG"

if [[ "$1" == "info" ]]; then
  exit 0
fi

if [[ "$1" == "container" && "${2:-}" == "inspect" ]]; then
  case "$*" in
    *'{{.Config.Image}}'*)
      printf '%s\n' "$KEYCLOAK_IMAGE"
      ;;
    *'{{json .HostConfig.PortBindings}}'*)
      printf '%s\n' \
        '{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]}'
      ;;
    *'{{.State.Running}}'*)
      printf 'false\n'
      ;;
  esac
  exit 0
fi

case "$1" in
  port)
    # Docker reports no active mapping while the retained container is not running.
    exit 0
    ;;
  start)
    exit 0
    ;;
  exec)
    if [[ "$*" == *'--interactive'* ]]; then
      cat >/dev/null
    fi
    exit 0
    ;;
esac

printf 'unexpected docker invocation: %s\n' "$*" >&2
exit 64
""",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "python3",
        '#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$@"\n',
    )
    _write_executable(
        fake_bin / "jq",
        r"""#!/usr/bin/env bash
set -euo pipefail

args="$*"
if [[ "$args" == *'--null-input'* ]]; then
  printf '{}\n'
  exit 0
fi

cat >/dev/null
if [[ "$args" == *'8080/tcp'* ]]; then
  printf '127.0.0.1:8080\n'
  exit 0
fi
if [[ "$args" == *'--exit-status'* ]]; then
  exit 1
fi
""",
    )

    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "DOCKER_CALL_LOG": str(docker_calls),
        "KEYCLOAK_IMAGE": "example.invalid/keycloak:1",
        "REAL_PYTHON": sys.executable,
    }
    env["KC_BOOTSTRAP_ADMIN_PASSWORD"] = "<KC_BOOTSTRAP_ADMIN_PASSWORD>"

    completed = subprocess.run(
        ["bash", str(START_KEYCLOAK)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    calls = docker_calls.read_text().splitlines()
    assert any(".HostConfig.PortBindings" in call for call in calls)
    assert "start langchain-keycloak" in calls
    assert not any(call.startswith("port ") for call in calls)


def test_new_realm_refreshes_auth_and_uses_target_realm(
    tmp_path: Path,
) -> None:
    """A new realm needs a fresh token and explicit administration target."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    _write_executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "$1" == "info" ]]; then
  exit 0
fi

if [[ "$1" == "container" && "${2:-}" == "inspect" ]]; then
  case "$*" in
    *'{{.Config.Image}}'*)
      printf '%s\n' "$KEYCLOAK_IMAGE"
      ;;
    *'{{json .HostConfig.PortBindings}}'*)
      printf '%s\n' \
        '{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]}'
      ;;
    *'{{.State.Running}}'*)
      printf 'true\n'
      ;;
  esac
  exit 0
fi

if [[ "$1" != "exec" ]]; then
  printf 'unexpected docker invocation\n' >&2
  exit 64
fi

args="$*"
if [[ "$args" == *'config credentials'* ]]; then
  auth_count=0
  if [[ -f "$AUTH_STATE_DIR/auth-count" ]]; then
    read -r auth_count < "$AUTH_STATE_DIR/auth-count"
  fi
  printf '%s\n' "$((auth_count + 1))" > "$AUTH_STATE_DIR/auth-count"
  printf 'authenticate\n' >> "$AUTH_STATE_DIR/events"
  exit 0
fi

if [[ "$args" == *' --realm langchain-agent'* ]]; then
  printf 'HTTP 401 Unauthorized\n' >&2
  exit 1
fi

if [[ "$args" == *' get realms/langchain-agent'* ]]; then
  [[ -f "$AUTH_STATE_DIR/realm-created" ]]
  exit
fi

if [[ "$args" == *' create realms '* ]]; then
  : > "$AUTH_STATE_DIR/realm-created"
  printf 'create-realm\n' >> "$AUTH_STATE_DIR/events"
  exit 0
fi

if [[ -f "$AUTH_STATE_DIR/realm-created" \
      && "$args" != *' --target-realm langchain-agent'* ]]; then
  printf 'missing --target-realm on realm-scoped call: %s\n' "$args" >&2
  exit 65
fi

if [[ -f "$AUTH_STATE_DIR/realm-created" \
      && "$args" == *' --target-realm langchain-agent'* ]]; then
  read -r auth_count < "$AUTH_STATE_DIR/auth-count"
  if (( auth_count < 2 )); then
    printf 'HTTP 401 Unauthorized\n' >&2
    exit 1
  fi
fi

if [[ -f "$AUTH_STATE_DIR/realm-created" \
      && "$args" == *' --target-realm langchain-agent'* \
      && ! -f "$AUTH_STATE_DIR/first-target-realm-call" ]]; then
  : > "$AUTH_STATE_DIR/first-target-realm-call"
  printf 'target-realm-call\n' >> "$AUTH_STATE_DIR/events"
fi

if [[ "$args" == *'--interactive'* ]]; then
  cat >/dev/null
fi
if [[ "$args" == *' --id'* ]]; then
  printf 'test-resource-id\n'
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "python3",
        "#!/usr/bin/env bash\nprintf '{}\\n'\n",
    )
    _write_executable(
        fake_bin / "jq",
        r"""#!/usr/bin/env bash
set -euo pipefail

args="$*"
if [[ "$args" == *'--null-input'* ]]; then
  printf '{}\n'
  exit 0
fi

cat >/dev/null
if [[ "$args" == *'8080/tcp'* ]]; then
  printf '127.0.0.1:8080\n'
  exit 0
fi
if [[ "$args" == *'--exit-status'* ]]; then
  exit 1
fi
""",
    )

    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AUTH_STATE_DIR": str(state_dir),
        "KEYCLOAK_IMAGE": "example.invalid/keycloak:1",
    }
    env["KC_BOOTSTRAP_ADMIN_PASSWORD"] = "<KC_BOOTSTRAP_ADMIN_PASSWORD>"

    completed = subprocess.run(
        ["bash", str(START_KEYCLOAK)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert (state_dir / "events").read_text().splitlines() == [
        "authenticate",
        "create-realm",
        "authenticate",
        "target-realm-call",
    ]


def test_admin_login_401_explains_retained_container_password(
    tmp_path: Path,
) -> None:
    """A retained container keeps its existing admin credential."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    _write_executable(
        fake_bin / "docker",
        r"""#!/usr/bin/env bash
set -euo pipefail

if [[ "$1" == "info" ]]; then
  exit 0
fi

if [[ "$1" == "container" && "${2:-}" == "inspect" ]]; then
  case "$*" in
    *'{{.Config.Image}}'*)
      printf '%s\n' "$KEYCLOAK_IMAGE"
      ;;
    *'{{json .HostConfig.PortBindings}}'*)
      printf '%s\n' \
        '{"8080/tcp":[{"HostIp":"127.0.0.1","HostPort":"8080"}]}'
      ;;
    *'{{.State.Running}}'*)
      printf 'true\n'
      ;;
  esac
  exit 0
fi

if [[ "$1" == "exec" && "$*" == *'config credentials'* ]]; then
  printf 'HTTP 401 Unauthorized\n' >&2
  exit 1
fi

printf 'unexpected docker invocation: %s\n' "$*" >&2
exit 64
""",
    )
    _write_executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\nexit 0\n",
    )
    _write_executable(
        fake_bin / "python3",
        "#!/usr/bin/env bash\nprintf '{}\\n'\n",
    )
    _write_executable(
        fake_bin / "jq",
        r"""#!/usr/bin/env bash
set -euo pipefail

cat >/dev/null
printf '127.0.0.1:8080\n'
""",
    )

    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "KEYCLOAK_IMAGE": "example.invalid/keycloak:1",
        "KC_BOOTSTRAP_ADMIN_PASSWORD": "<KC_BOOTSTRAP_ADMIN_PASSWORD>",
    }

    completed = subprocess.run(
        ["bash", str(START_KEYCLOAK)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "KC_BOOTSTRAP_ADMIN_PASSWORD" in completed.stderr
    assert "retained container langchain-keycloak" in completed.stderr
    assert "HTTP 401 Unauthorized" not in completed.stderr
