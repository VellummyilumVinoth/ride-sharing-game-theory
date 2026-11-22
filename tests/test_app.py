"""Smoke test: the whole dashboard renders without an exception."""
from streamlit.testing.v1 import AppTest


def test_app_renders():
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.tabs) == 7
