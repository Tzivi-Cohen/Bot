"""User-facing error messages for OpenAI failures, with secrets redacted."""

import os
import re

# OpenAI keys look like "sk-..." / "sk-proj-...".
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*]{6,}")


def redact_secrets(text: str) -> str:
    """Remove the configured API key (and anything key-shaped) from text."""
    text = str(text)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        text = text.replace(key, "***")
    return _KEY_PATTERN.sub("sk-***", text)


def _is_ssl_verification_error(error: BaseException) -> bool:
    """True if the error (or anything that caused it) is a certificate failure."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            return True
        error = error.__cause__ or error.__context__
    return False


def friendly_api_error(error: Exception) -> str:
    """Hebrew message for an API error. Never includes the key."""
    if _is_ssl_verification_error(error):
        return (
            "אימות אישור האבטחה (SSL) מול OpenAI נכשל. בדרך כלל הסיבה היא תוכנת "
            "אנטי־וירוס או סינון רשת שבודקים תעבורת HTTPS. ראו את הסעיף "
            "'פתרון בעיות' ב־README."
        )
    name = type(error).__name__
    if name == "AuthenticationError":
        return "מפתח ה־API של OpenAI אינו תקין. יש לבדוק את הקובץ ‎.env."
    if name == "RateLimitError":
        return "חריגה ממגבלת השימוש או מהמכסה של OpenAI. נסי שוב מאוחר יותר."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError"):
        return "אין חיבור לשרת של OpenAI. יש לבדוק את החיבור לאינטרנט."
    return f"שגיאה בפנייה ל־OpenAI: {redact_secrets(error)}"
