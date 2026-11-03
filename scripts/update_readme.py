"""Copy generated tables from reports/results.md into the marked blocks of README.md."""

from secom.config import PROJECT_ROOT, REPORTS_DIR
from secom.reporting import RESULTS_MARKDOWN, inject_sections, parse_sections

README_PATH = PROJECT_ROOT / "README.md"


def main() -> None:
    sections = parse_sections((REPORTS_DIR / RESULTS_MARKDOWN).read_text(encoding="utf-8"))
    readme = README_PATH.read_text(encoding="utf-8")
    README_PATH.write_text(inject_sections(readme, sections), encoding="utf-8")


if __name__ == "__main__":
    main()
