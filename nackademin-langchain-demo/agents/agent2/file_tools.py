from pathlib import Path
from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ROOTS = {
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "mlops": Path.home() / "MLOps",
    "project": PROJECT_ROOT,
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".log"
}

MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB


def _allowed_roots_text() -> str:
    lines = []
    for alias, path in ALLOWED_ROOTS.items():
        exists = "finns" if path.exists() else "saknas"
        lines.append(f"- {alias}: {path} ({exists})")
    return "\n".join(lines)


def _resolve_root(root_alias: str) -> Path:
    root = ALLOWED_ROOTS.get(root_alias.lower())
    if not root:
        raise ValueError(
            f"Ogiltig root_alias: {root_alias}. Tillåtna värden är: {', '.join(ALLOWED_ROOTS.keys())}"
        )
    return root.resolve()


def _is_path_allowed(path: Path) -> bool:
    resolved = path.resolve()
    for root in ALLOWED_ROOTS.values():
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _safe_read_text_file(path: Path) -> str:
    if not path.exists():
        return "Filen finns inte."

    if not path.is_file():
        return "Sökvägen är inte en fil."

    if not _is_path_allowed(path):
        return "Åtkomst nekad. Filen ligger utanför tillåtna mappar."

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return f"Filtypen {path.suffix} stöds inte i denna version."

    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        return "Filen är för stor för att läsas i denna version."

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            return f"Kunde inte läsa filen som text: {e}"
    except Exception as e:
        return f"Ett fel uppstod vid läsning av filen: {e}"


@tool
def list_allowed_roots() -> str:
    """Visa vilka mappar agenten får söka i."""
    return "Tillåtna mappar:\n" + _allowed_roots_text()


@tool
def search_files_by_name(keyword: str, root_alias: str = "documents", max_results: int = 10) -> str:
    """
    Sök efter filer vars filnamn innehåller ett visst ord i en tillåten mapp.
    root_alias kan vara: documents, downloads, project
    """
    try:
        root = _resolve_root(root_alias)
    except ValueError as e:
        return str(e)

    if not root.exists():
        return f"Mappen finns inte: {root}"

    keyword_lower = keyword.lower()
    matches = []

    for path in root.rglob("*"):
        if path.is_file() and keyword_lower in path.name.lower():
            matches.append(path)

    if not matches:
        return f"Inga filer hittades med '{keyword}' i {root}"

    matches = matches[:max_results]

    lines = [f"Hittade {len(matches)} filer:"]
    for i, path in enumerate(matches, start=1):
        lines.append(f"{i}. {path}")

    return "\n".join(lines)


@tool
def search_file_content(query: str, root_alias: str = "documents", max_results: int = 10) -> str:
    """
    Sök efter textinnehåll i textfiler i en tillåten mapp.
    root_alias kan vara: documents, downloads, project
    """
    try:
        root = _resolve_root(root_alias)
    except ValueError as e:
        return str(e)

    if not root.exists():
        return f"Mappen finns inte: {root}"

    query_lower = query.lower()
    results = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
        except Exception:
            continue

        content = _safe_read_text_file(path)
        if not content or content.startswith("Ett fel uppstod") or content.startswith("Kunde inte läsa"):
            continue

        if query_lower in content.lower():
            snippet_index = content.lower().find(query_lower)
            start = max(0, snippet_index - 120)
            end = min(len(content), snippet_index + 180)
            snippet = content[start:end].replace("\n", " ")
            results.append((path, snippet))

        if len(results) >= max_results:
            break

    if not results:
        return f"Inget innehåll med '{query}' hittades i {root}"

    lines = [f"Hittade {len(results)} träffar:"]
    for i, (path, snippet) in enumerate(results, start=1):
        lines.append(f"{i}. {path}\n   Utdrag: {snippet}")

    return "\n".join(lines)


@tool
def read_text_file(file_path: str) -> str:
    """Läs innehållet i en textfil från en tillåten mapp."""
    path = Path(file_path).expanduser()
    return _safe_read_text_file(path)

@tool
def list_recent_files(root_alias: str = "documents", max_results: int = 10) -> str:
    """Visa de senaste ändrade filerna i en tillåten mapp."""