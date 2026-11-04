"""Offline human-review UI tested with fictional text and typed input only."""
import csv
import io
from pathlib import Path
import socket

import pytest

from scripts import review_helper as helper


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("review must not make network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def review_file(tmp_path):
    text = tmp_path / "confirmation.txt"
    lines = [f"Ordinary line {i}" for i in range(1, 81)]
    lines[0] = "Trade Date: 2024-01-02"
    lines[9] = "Notional Amount USD 100,000,000"
    lines[64] = "Fixed Rate: 4.25%"
    lines[69] = "Floating Rate Option: USD-SOFR"
    lines[79] = "Termination Date: 2029-01-04"
    text.write_text("\n".join(lines), encoding="utf-8")
    columns = ["document_url", "era", "label", "page_count", "page_count_basis", "text_path", "is_confirmation", "notes"]
    rows = [dict(zip(columns, [f"https://www.sec.gov/fixture/{i}", era, label, "3", "estimated_chars", str(text), "", "retain me"]))
            for i, (era, label) in enumerate((("2001–2005", "likely_confirmation"), ("2022+", "maybe"), ("2022+", "unlikely")))]
    path = tmp_path / "manual_check.csv"
    write(path, columns, rows)
    return path, text, columns, rows


def write(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def answers(*values):
    iterator = iter(values)
    def response(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None
    return response


def test_first_sixty_plus_keyword_lines_highlighted_once(review_file):
    path, text, _, rows = review_file
    output = io.StringIO()
    helper.show_document(rows[0], path, output=output, color=False)
    result = output.getvalue()
    assert "URL: https://www.sec.gov/fixture/0" in result
    assert "Era: 2001–2005" in result and "Screen label: likely_confirmation" in result
    assert "Pages: 3 (estimated_chars)" in result
    assert "Ordinary line 60" in result and "Ordinary line 61" not in result
    for phrase in ("Notional", "Fixed Rate", "Floating Rate Option", "Termination Date", "Trade Date"):
        assert f"[[{phrase}]]" in result
    assert result.count("USD 100,000,000") == 1
    assert "   65 |" in result and "   80 |" in result
    color = io.StringIO()
    helper.show_document(rows[0], path, output=color, color=True)
    assert "\033[1;33mNotional\033[0m" in color.getvalue()


def test_each_answer_saved_before_next_prompt_and_summary(review_file):
    path, text, _, original = review_file
    source_before = text.read_bytes()
    index = 0
    def response(prompt):
        nonlocal index
        saved = read(path)
        if index >= 1:
            assert saved[0]["is_confirmation"] == "y" and saved[0]["product_type"] == "irs_fixed_float"
        if index >= 2:
            assert saved[1]["is_confirmation"] == "n" and saved[1]["product_type"] == "cap_floor"
        result = ("y 1", "n 3", "s")[index]
        index += 1
        return result
    output = io.StringIO()
    assert helper.review(path, input_fn=response, output=output) == 0
    saved = read(path)
    assert [row["is_confirmation"] for row in saved] == ["y", "n", ""]
    assert saved[2]["review_status"] == "skipped"
    assert all(row["notes"] == "retain me" for row in saved)
    assert text.read_bytes() == source_before
    summary = output.getvalue()
    assert "likely_confirmation: yes=1 no=0" in summary
    assert "maybe: yes=0 no=1" in summary
    assert "2001–2005: yes=1 no=0" in summary and "2022+: yes=0 no=1" in summary
    assert "skipped=1 unanswered=0" in summary


def test_eof_resume_and_revisit_skips(review_file):
    path = review_file[0]
    helper.review(path, input_fn=answers("y2", "s4"), output=io.StringIO())
    output = io.StringIO()
    helper.review(path, input_fn=answers("n"), output=output)
    assert "fixture/0" not in output.getvalue() and "fixture/1" not in output.getvalue()
    assert "fixture/2" in output.getvalue()
    assert read(path)[0]["product_type"] == "basis"
    helper.review(path, input_fn=answers("y 1"), output=io.StringIO(), revisit_skipped=True)
    saved = read(path)
    assert saved[1]["is_confirmation"] == "y" and saved[1]["product_type"] == "irs_fixed_float"
    assert saved[1]["review_status"] == "answered"


def test_invalid_input_and_keyboard_interrupt_preserve_progress(review_file):
    path = review_file[0]
    inputs = iter(["bad", "y 9", "y 1"])
    def response(prompt):
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt
    output = io.StringIO()
    assert helper.review(path, input_fn=response, output=output) == 130
    assert read(path)[0]["is_confirmation"] == "y"
    assert read(path)[1]["is_confirmation"] == ""
    assert output.getvalue().count("Enter y, n or s") == 2


def test_missing_text_is_not_labelled_or_downloaded(review_file):
    path, _, columns, rows = review_file
    rows[0]["text_path"] = "https://www.sec.gov/not-local.txt"
    rows[1]["text_path"] = "/missing/local/file.txt"
    write(path, columns, rows)
    output = io.StringIO()
    helper.review(path, input_fn=answers("n 4"), output=output)
    assert [row["is_confirmation"] for row in read(path)] == ["", "", "n"]
    assert output.getvalue().count("Text unavailable; left unanswered") == 2


def test_existing_answers_and_other_columns_are_preserved(review_file):
    path, _, columns, rows = review_file
    rows[0]["is_confirmation"] = "TRUE"
    write(path, columns, rows)
    helper.review(path, input_fn=answers("n", "s"), output=io.StringIO())
    assert read(path)[0]["is_confirmation"] == "TRUE"
    assert read(path)[0]["notes"] == "retain me"


def test_concurrent_edit_refuses_overwrite(review_file):
    path, _, columns, rows = review_file
    def response(prompt):
        rows[0]["notes"] = "another reviewer edited this"
        write(path, columns, rows)
        return "y"
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="changed on disk"):
        helper.review(path, input_fn=response, output=output)
    assert read(path)[0]["notes"] == "another reviewer edited this"
    assert read(path)[0]["is_confirmation"] == ""
    assert "yes=0 no=0" in output.getvalue()


def test_relative_paths_and_terminal_escape_safety(review_file, monkeypatch):
    path, text, _, rows = review_file
    monkeypatch.chdir(path.parent)
    assert helper.local_text_path(text.name, path) == text.resolve()
    text.write_text("Notional \x1b]52;c;hidden\x07", encoding="utf-8")
    output = io.StringIO()
    helper.show_document(rows[0], path, output=output, color=False)
    assert "\x1b" not in output.getvalue() and "\\x1b" in output.getvalue()


@pytest.mark.parametrize("value,expected", [("y", ("y", None)), ("N4", ("n", "other")), ("s,2", ("s", "basis"))])
def test_answer_keys(value, expected):
    assert helper.parse_answer(value) == expected


def test_cli_no_color_and_eof(review_file, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", answers("y 1"))
    assert helper.main(["--csv", str(review_file[0]), "--no-color"]) == 0
    assert "[[Notional]]" in capsys.readouterr().out
