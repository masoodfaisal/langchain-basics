#!/usr/bin/env bash

set -euo pipefail

required_variables=(OPENAI_API_KEY)

for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    printf 'Error: %s is not available in this shell environment.\n' \
      "$variable_name" >&2
    printf 'Export it in the parent shell, then run this script again.\n' >&2
    exit 1
  fi
done

# Explicitly pass the existing values to the LangGraph child process. Never
# print them, pass them as command-line arguments, or copy them into .env.
export OPENAI_API_KEY

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_file="$project_dir/langgraph.json"

if [[ -x "$project_dir/.venv/bin/langgraph" ]]; then
  langgraph_bin="$project_dir/.venv/bin/langgraph"
elif command -v langgraph >/dev/null 2>&1; then
  langgraph_bin="$(command -v langgraph)"
else
  printf 'Error: langgraph was not found in .venv or PATH.\n' >&2
  exit 1
fi

cd "$project_dir"
exec "$langgraph_bin" dev --config "$config_file" "$@"
