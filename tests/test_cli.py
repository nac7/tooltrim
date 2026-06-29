import json

from tooltrim.cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "tooltrim" in capsys.readouterr().out


def test_demo_runs_and_keeps_needles(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "smaller" in out
    assert "NO" not in out  # every needle kept


def test_compress_file_query_aware(tmp_path, capsys):
    blob = json.dumps([{"id": i, "note": f"row {i}"} for i in range(400)]
                      + [{"id": 999, "note": "refund to customer 4417"}])
    f = tmp_path / "big.json"
    f.write_text(blob, encoding="utf-8")

    rc = main(["compress", str(f), "-q", "refund 4417", "-m", "80", "--stats"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "4417" in captured.out          # needle survived to stdout
    assert "saved" in captured.err         # stats went to stderr
    assert len(captured.out) < len(blob)


def test_no_subcommand_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()
