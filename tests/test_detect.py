import json

from tooltrim.detect import detect_type


def test_detect_json_object():
    assert detect_type('{"a": 1, "b": [1,2,3]}') == "json"


def test_detect_json_array():
    assert detect_type(json.dumps([{"x": 1}, {"x": 2}])) == "json"


def test_detect_html():
    html = "<html><body><div>hello</div><p>world</p></body></html>"
    assert detect_type(html) == "html"


def test_detect_html_fragment_by_tag_density():
    frag = "<div><span>a</span><span>b</span><ul><li>x</li><li>y</li></ul></div>"
    assert detect_type(frag) == "html"


def test_detect_logs():
    logs = "\n".join(
        f"2026-06-27 10:00:0{i} INFO starting worker {i}" for i in range(6)
    )
    assert detect_type(logs) == "logs"


def test_detect_tabular_csv():
    csv = "name,age,city\nalice,30,nyc\nbob,25,sf\ncarol,41,la"
    assert detect_type(csv) == "tabular"


def test_detect_text_fallback():
    assert detect_type("just a sentence with no structure at all") == "text"


def test_detect_empty():
    assert detect_type("") == "text"
    assert detect_type("   \n  ") == "text"
