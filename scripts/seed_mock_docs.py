"""Create small sample documents for local demos."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "sample_docs"


def main() -> None:
    """Write starter documents if they do not already exist."""

    SAMPLE_DOCS.mkdir(exist_ok=True)

    docs = {
        "enterprise_ai_policy.md": (
            "# Enterprise AI Policy\n\n"
            "Use approved models, protect confidential data, and keep humans in "
            "the loop for high-impact decisions.\n"
        ),
        "sales_playbook.md": (
            "# Sales Playbook\n\n"
            "Qualify customer needs, summarize business value, and document next "
            "steps after each executive conversation.\n"
        ),
    }

    for filename, content in docs.items():
        path = SAMPLE_DOCS / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(f"Created {path}")
        else:
            print(f"Skipped existing {path}")


if __name__ == "__main__":
    main()
