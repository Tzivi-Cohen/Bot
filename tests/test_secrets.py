"""The API key is read from .env and never appears in user-facing messages."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rag.errors import friendly_api_error, redact_secrets
from tests.conftest import AuthenticationError

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-proj-FAKEFAKEFAKE0123456789abcdef"


def test_key_is_read_from_dotenv_regardless_of_working_dir(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "config.py", project / "config.py")
    (project / ".env").write_text(f"OPENAI_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import config; "
        "print(config.has_api_key())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(project)],
        cwd=elsewhere, env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "True"
    assert FAKE_KEY not in out.stdout + out.stderr


def test_redact_configured_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    assert FAKE_KEY not in redact_secrets(f"error with {FAKE_KEY} inside")


def test_redact_key_shaped_strings_even_if_not_configured():
    assert "abcdef123456" not in redact_secrets("bad key sk-abcdef123456XYZ")


def test_friendly_error_hides_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    message = friendly_api_error(RuntimeError(f"boom {FAKE_KEY}"))
    assert FAKE_KEY not in message
    auth = friendly_api_error(AuthenticationError(f"Incorrect API key {FAKE_KEY}"))
    assert FAKE_KEY not in auth and "מפתח ה־API" in auth


def test_env_files_ignored_by_git_and_example_has_no_secret():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in example
    value = example.split("OPENAI_API_KEY=", 1)[1].splitlines()[0].strip()
    assert value == ""


def test_no_key_hardcoded_in_source():
    import re

    real_key = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
    sources = list(ROOT.glob("*.py")) + list((ROOT / "rag").glob("*.py"))
    sources += [ROOT / "README.md", ROOT / ".env.example", ROOT / "requirements.txt"]
    for path in sources:
        if path.exists():
            assert not real_key.search(path.read_text(encoding="utf-8")), path


def test_ssl_interception_gets_specific_message():
    class APIConnectionError(Exception):
        pass

    try:
        try:
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        except OSError as inner:
            raise APIConnectionError("Connection error.") from inner
    except APIConnectionError as outer:
        message = friendly_api_error(outer)
    assert "SSL" in message and "אנטי־וירוס" in message
