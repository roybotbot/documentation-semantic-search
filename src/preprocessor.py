"""Markdown preprocessing to clean artifacts before chunking."""
import re


def strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by --- at the start of a file."""
    return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)


def strip_mdx_components(text: str) -> str:
    """Remove JSX/MDX component tags like <Tabs>, <TabItem>, <DocCardList />.

    Keeps the text content between opening and closing tags.
    """
    # Self-closing tags: <Component /> or <Component prop="val" />
    text = re.sub(r"<[A-Z][a-zA-Z]*\b[^>]*/\s*>", "", text)
    # Opening/closing tags: <Component ...> and </Component>
    text = re.sub(r"</?[A-Z][a-zA-Z]*\b[^>]*>", "", text)
    return text


def strip_import_statements(text: str) -> str:
    """Remove ES-style import lines common in MDX files."""
    return re.sub(r"^import\s+.*$", "", text, flags=re.MULTILINE)


def normalize_admonitions(text: str) -> str:
    """Convert Docusaurus admonition syntax to plain text.

    Turns :::note, :::tip, :::warning etc. into readable labels.
    """
    text = re.sub(r"^:::\s*(\w+)\s*$", r"[\1]", text, flags=re.MULTILINE)
    text = re.sub(r"^:::\s*$", "", text, flags=re.MULTILINE)
    return text


def collapse_whitespace(text: str) -> str:
    """Reduce runs of 3+ blank lines to 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def preprocess(text: str) -> str:
    """Run all preprocessing steps on raw markdown text."""
    text = strip_frontmatter(text)
    text = strip_import_statements(text)
    text = strip_mdx_components(text)
    text = normalize_admonitions(text)
    text = collapse_whitespace(text)
    return text.strip()
