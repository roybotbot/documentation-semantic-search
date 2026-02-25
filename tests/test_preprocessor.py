"""Tests for markdown preprocessing."""
from src.preprocessor import (
    strip_frontmatter,
    strip_mdx_components,
    strip_import_statements,
    normalize_admonitions,
    collapse_whitespace,
    preprocess,
)


def test_strip_frontmatter_removes_yaml():
    text = "---\ntitle: Test\nsidebar: docs\n---\n# Hello"
    assert strip_frontmatter(text) == "# Hello"


def test_strip_frontmatter_ignores_missing():
    text = "# No frontmatter here"
    assert strip_frontmatter(text) == text


def test_strip_frontmatter_only_removes_first():
    text = "---\ntitle: Test\n---\nContent\n---\nNot frontmatter\n---"
    result = strip_frontmatter(text)
    assert "Content" in result
    assert "Not frontmatter" in result


def test_strip_mdx_self_closing():
    text = "Before\n<DocCardList />\nAfter"
    assert strip_mdx_components(text) == "Before\n\nAfter"


def test_strip_mdx_paired_tags():
    text = "<Tabs>\n<TabItem value='a'>\nContent\n</TabItem>\n</Tabs>"
    result = strip_mdx_components(text)
    assert "Content" in result
    assert "<Tabs>" not in result
    assert "<TabItem" not in result


def test_strip_mdx_leaves_html_alone():
    text = "<div>regular html</div>\n<code>also fine</code>"
    assert strip_mdx_components(text) == text


def test_strip_import_statements():
    text = "import Tabs from '@theme/Tabs';\nimport TabItem from '@theme/TabItem';\n\n# Title"
    result = strip_import_statements(text)
    assert "import" not in result
    assert "# Title" in result


def test_normalize_admonitions():
    text = ":::note\nThis is a note.\n:::"
    result = normalize_admonitions(text)
    assert "[note]" in result
    assert "This is a note." in result
    assert ":::" not in result


def test_collapse_whitespace():
    text = "Line 1\n\n\n\n\nLine 2"
    assert collapse_whitespace(text) == "Line 1\n\nLine 2"


def test_preprocess_handles_real_mdx():
    """Full pipeline on a realistic MDX file."""
    text = """---
title: Farming guide
sidebar_position: 1
---

import Tabs from '@theme/Tabs';

# Getting started

<Tabs>
<TabItem value="cli">

Run the following command:

```bash
chia start farmer
```

</TabItem>
</Tabs>

:::warning
Make sure your node is synced first.
:::
"""
    result = preprocess(text)
    assert "title:" not in result
    assert "import " not in result
    assert "<Tabs>" not in result
    assert "Run the following command:" in result
    assert "chia start farmer" in result
    assert "[warning]" in result
    assert "Make sure your node is synced first." in result
