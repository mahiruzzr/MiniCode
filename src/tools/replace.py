"""Fuzzy string-replacement engine used by the edit and apply_patch tools.

Exact-string edits often fail on whitespace, indentation, escaping, or
line-ending drift. This module tries an ordered ladder of replacement
strategies until one produces a candidate that exists in the content, is a
plausible size, and (unless replacing all occurrences) is unique.

A replacer is a generator ``(content, find) -> Iterator[str]`` that yields
candidate substrings of ``content``. ``replace`` validates each candidate, so
replacers only propose matches and do not handle uniqueness or size limits
themselves.

The module contains pure functions with no I/O, which keeps the ladder
testable in isolation.
"""

from __future__ import annotations

from collections.abc import Iterator

# Minimum block-anchor similarity for a lone candidate, and for the
# best-scoring candidate when several exist.
_SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.65
_MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.65


class ReplaceError(ValueError):
    """Base class for replacement failures returned to the model as errors.

    Subclasses distinguish a missing match, an ambiguous match, and a match
    that is implausibly larger than the requested text.
    """


class NotFoundError(ReplaceError):
    pass


class MultipleMatchesError(ReplaceError):
    pass


class DisproportionateMatchError(ReplaceError):
    pass


def _levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between ``a`` and ``b``."""
    if a == "" or b == "":
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def _span(lines: list[str], start: int, end: int) -> str:
    """Return half-open character offsets covering lines ``start..end``.

    The line range is inclusive, and the returned range is half-open, so
    slicing the newline-joined content yields exactly those lines.
    """
    start_index = sum(len(lines[k]) + 1 for k in range(start))
    end_index = start_index
    for k in range(start, end + 1):
        end_index += len(lines[k])
        if k < end:
            end_index += 1
    return start_index, end_index


# --- replacer ladder, tried in order ---------------------------------------


def simple_replacer(content: str, find: str) -> Iterator[str]:
    """Yield ``find`` unchanged: the exact-match strategy."""
    yield find


def line_trimmed_replacer(content: str, find: str) -> Iterator[str]:
    """Match a line-aligned block after trimming each line.

    A trailing newline in ``find`` is ignored, and the remaining lines must
    match one-for-one.
    """
    original = content.split("\n")
    search = find.split("\n")
    if search and search[-1] == "":
        search.pop()
    for i in range(len(original) - len(search) + 1):
        if all(
            original[i + j].strip() == search[j].strip() for j in range(len(search))
        ):
            s, e = _span(original, i, i + len(search) - 1)
            yield content[s:e]


def block_anchor_replacer(content: str, find: str) -> Iterator[str]:
    """Match a block by anchoring on its first and last trimmed lines.

    Intended for blocks of three or more lines. Candidate line counts may
    drift by at most a quarter of the requested size (minimum one line), and
    interior lines are scored by normalized Levenshtein similarity. A lone
    candidate, or the best of several, must reach the corresponding threshold.
    """
    original = content.split("\n")
    search = find.split("\n")
    if len(search) < 3:
        return
    if search and search[-1] == "":
        search.pop()

    first = search[0].strip()
    last = search[-1].strip()
    block_size = len(search)
    max_delta = max(1, block_size // 4)

    candidates: list[tuple[int, int]] = []
    for i in range(len(original)):
        if original[i].strip() != first:
            continue
        for j in range(i + 2, len(original)):
            if original[j].strip() == last:
                if abs((j - i + 1) - block_size) <= max_delta:
                    candidates.append((i, j))
                break
    if not candidates:
        return

    def _similarity(start: int, end: int) -> float:
        actual = end - start + 1
        to_check = min(block_size - 2, actual - 2)
        if to_check <= 0:
            return 1.0
        sim = 0.0
        for j in range(1, min(block_size - 1, actual - 1)):
            orig = original[start + j].strip()
            srch = search[j].strip()
            max_len = max(len(orig), len(srch))
            if max_len == 0:
                continue
            sim += (1 - _levenshtein(orig, srch) / max_len) / to_check
        return sim

    if len(candidates) == 1:
        start, end = candidates[0]
        if _similarity(start, end) >= _SINGLE_CANDIDATE_SIMILARITY_THRESHOLD:
            s, e = _span(original, start, end)
            yield content[s:e]
        return

    best = None
    best_sim = -1.0
    for start, end in candidates:
        # Compare every candidate and keep the highest average similarity.
        actual = end - start + 1
        to_check = min(block_size - 2, actual - 2)
        if to_check <= 0:
            sim = 1.0
        else:
            total = 0.0
            for j in range(1, min(block_size - 1, actual - 1)):
                orig = original[start + j].strip()
                srch = search[j].strip()
                max_len = max(len(orig), len(srch))
                if max_len == 0:
                    continue
                total += 1 - _levenshtein(orig, srch) / max_len
            sim = total / to_check
        if sim > best_sim:
            best_sim = sim
            best = (start, end)
    if best is not None and best_sim >= _MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD:
        s, e = _span(original, best[0], best[1])
        yield content[s:e]


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs, including newlines, and trim the ends."""
    import re

    return re.sub(r"\s+", " ", text).strip()


def whitespace_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """Match text while ignoring differences in whitespace.

    Whitespace runs are collapsed when comparing a single line or a
    line-aligned block. A single-line substring can also be located by
    matching the words of ``find`` against arbitrary whitespace.
    """
    import re

    normalized_find = _normalize_ws(find)
    lines = content.split("\n")
    for line in lines:
        if _normalize_ws(line) == normalized_find:
            yield line
        elif normalized_find in _normalize_ws(line):
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                try:
                    m = re.search(pattern, line)
                    if m:
                        yield m.group(0)
                except re.error:
                    pass
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if _normalize_ws(block) == normalized_find:
                yield block


def indentation_flexible_replacer(content: str, find: str) -> Iterator[str]:
    """Match a line-aligned block after removing common leading indentation.

    Both ``find`` and each candidate block are dedented by their minimum
    non-empty indentation; the line count and all remaining text must match
    exactly.
    """
    import re

    def _dedent(text: str) -> str:
        lines = text.split("\n")
        non_empty = [ln for ln in lines if ln.strip()]
        if not non_empty:
            return text
        min_indent = min(len(re.match(r"^(\s*)", ln).group(1)) for ln in non_empty)
        return "\n".join(ln if not ln.strip() else ln[min_indent:] for ln in lines)

    normalized_find = _dedent(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if _dedent(block) == normalized_find:
            yield block


def escape_normalized_replacer(content: str, find: str) -> Iterator[str]:
    """Match text after interpreting backslash escapes in ``find``.

    The unescaped text is tried literally, then compared against line-aligned
    blocks that are also decoded. This lets ``\\n`` written as two characters
    match a real newline in the content.
    """
    import re

    def _unescape(s: str) -> str:
        # Keys match the regex alternatives; self-mappings drop the backslash.
        mapping = {
            "n": "\n",
            "t": "\t",
            "r": "\r",
            "'": "'",
            '"': '"',
            "`": "`",
            "\\": "\\",
            "\n": "\n",
            "$": "$",
        }
        return re.sub(
            r"\\(n|t|r|'|\"|`|\\|\n|\$)",
            lambda m: mapping.get(m.group(1), m.group(0)),
            s,
        )

    unescaped_find = _unescape(find)
    if unescaped_find in content:
        yield unescaped_find
    lines = content.split("\n")
    find_lines = unescaped_find.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if _unescape(block) == unescaped_find:
            yield block


def trimmed_boundary_replacer(content: str, find: str) -> Iterator[str]:
    """Match text after stripping whitespace from the outer boundaries.

    Only whitespace at the start and end of the match is ignored; whitespace
    inside the match must still be exact.
    """
    trimmed = find.strip()
    if trimmed == find:
        return
    if trimmed in content:
        yield trimmed
    lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed:
            yield block


def context_aware_replacer(content: str, find: str) -> Iterator[str]:
    """Match a same-length block by its boundaries and interior agreement.

    The first and last lines must match after stripping. At least half of the
    interior line pairs that are not both blank must also match exactly. This
    tolerates limited interior edits after stricter strategies fail.
    """
    find_lines = find.split("\n")
    if len(find_lines) < 3:
        return
    if find_lines and find_lines[-1] == "":
        find_lines.pop()
    content_lines = content.split("\n")
    first = find_lines[0].strip()
    last = find_lines[-1].strip()
    for i in range(len(content_lines)):
        if content_lines[i].strip() != first:
            continue
        for j in range(i + 2, len(content_lines)):
            if content_lines[j].strip() == last:
                block_lines = content_lines[i : j + 1]
                if len(block_lines) == len(find_lines):
                    matching = 0
                    total = 0
                    for k in range(1, len(block_lines) - 1):
                        bl = block_lines[k].strip()
                        fl = find_lines[k].strip()
                        if bl or fl:
                            total += 1
                            if bl == fl:
                                matching += 1
                    if total == 0 or matching / total >= 0.5:
                        yield "\n".join(block_lines)
                        break
                break


def multi_occurrence_replacer(content: str, find: str) -> Iterator[str]:
    """Yield ``find`` once per non-overlapping literal occurrence.

    Assumes ``find`` is non-empty; ``replace`` rejects empty searches before
    running the ladder.
    """
    start = 0
    while True:
        index = content.find(find, start)
        if index == -1:
            break
        yield find
        start = index + len(find)


_REPLACERS = (
    simple_replacer,
    line_trimmed_replacer,
    block_anchor_replacer,
    whitespace_normalized_replacer,
    indentation_flexible_replacer,
    escape_normalized_replacer,
    trimmed_boundary_replacer,
    context_aware_replacer,
    multi_occurrence_replacer,
)


def _is_disproportionate(search: str, old_string: str) -> bool:
    """Return True when ``search`` is implausibly larger than ``old_string``.

    The line-count guard rejects candidates with at least
    ``max(old_lines + 3, old_lines * 2)`` lines. Multi-line searches also
    reject candidates whose stripped length exceeds
    ``max(old_stripped_len + 500, old_stripped_len * 4)``.
    """
    old_lines = len(old_string.split("\n"))
    search_lines = len(search.split("\n"))
    if search_lines >= max(old_lines + 3, old_lines * 2):
        return True
    if old_lines == 1:
        return False
    return len(search.strip()) > max(
        len(old_string.strip()) + 500, len(old_string.strip()) * 4
    )


def replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Return ``content`` with ``old_string`` replaced by ``new_string``.

    Replacers are tried in order. Candidates that do not occur in ``content``
    are skipped, while an occurring candidate that is disproportionately
    larger than ``old_string`` raises immediately. Unless ``replace_all`` is
    set, the candidate must also be unique; otherwise the next candidate or
    strategy is tried. With ``replace_all``, the first accepted candidate is
    passed to ``str.replace`` to replace every occurrence.

    Raises:
        ValueError: ``old_string`` is empty or identical to ``new_string``.
        NotFoundError: No yielded candidate occurs in ``content``.
        MultipleMatchesError: No candidate is unique and ``replace_all`` is
            false.
        DisproportionateMatchError: An occurring candidate is far larger than
            ``old_string``.
    """
    if old_string == new_string:
        raise ValueError(
            "No changes to apply: old_string and new_string are identical."
        )
    if old_string == "":
        raise ValueError(
            "old_string cannot be empty when editing an existing file. "
            "Provide the exact text to replace, or use write for a full-file "
            "replacement."
        )

    found = False
    for replacer in _REPLACERS:
        for search in replacer(content, old_string):
            index = content.find(search)
            if index == -1:
                continue
            found = True
            if _is_disproportionate(search, old_string):
                raise DisproportionateMatchError(
                    "Refusing replacement because the matched span is much "
                    "larger than old_string. Re-read the file and provide the "
                    "full exact old_string for the intended replacement."
                )
            if replace_all:
                return content.replace(search, new_string)
            if index != content.rfind(search):
                continue  # not unique; try the next candidate/strategy
            return content[:index] + new_string + content[index + len(search) :]

    if not found:
        raise NotFoundError(
            "Could not find old_string in the file. It must match exactly, "
            "including whitespace, indentation, and line endings."
        )
    raise MultipleMatchesError(
        "Found multiple matches for old_string. Provide more surrounding "
        "context to make the match unique."
    )
