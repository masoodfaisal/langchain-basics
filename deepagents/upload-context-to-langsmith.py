from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

from deepagents.backends.context_hub import ContextHubBackend

hub = ContextHubBackend("-/research-agent-context")

results = hub.upload_files([
    (
        "/AGENTS.md",
        b"""# Research Agent Instructions

## Workflow

1. Break the request into research topics.
2. Delegate each topic to the research subagent.
3. Prefer primary and authoritative sources.
4. Include URLs for factual claims.
5. Clearly identify uncertainty.

## Additional context

Read `/context/policies/citations.md` before producing the final answer.
""",
    ),
    (
        "/policies/citations.md",
        b"""# Citation Policy

- Prefer official documentation and primary sources.
- Do not invent citations.
- Include the source URL.
- Distinguish sourced facts from conclusions.
""",
    ),
])

for result in results:
    print(result)