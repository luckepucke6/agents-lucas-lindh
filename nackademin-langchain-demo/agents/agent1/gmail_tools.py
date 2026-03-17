import base64
import os
from pathlib import Path

from langchain_core.tools import tool
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"


def get_gmail_service():
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def _get_header(headers, name):
    for header in headers:
        if header["name"].lower() == name.lower():
            return header["value"]
    return ""


def _extract_plain_text(payload):
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")

    if mime_type == "text/plain" and data:
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    for part in payload.get("parts", []):
        text = _extract_plain_text(part)
        if text:
            return text

    return ""


@tool
def search_emails(query: str, max_results: int = 5) -> str:
    """
    Sök i Gmail med Gmail-syntax, t.ex.:
    from:läraren@example.com is:unread
    newer_than:7d
    label:inbox
    """
    service = get_gmail_service()

    results = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )

    messages = results.get("messages", [])
    if not messages:
        return "Inga mejl hittades."

    output = []
    for msg in messages:
        full_msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg["id"], format="full")
            .execute()
        )

        headers = full_msg.get("payload", {}).get("headers", [])
        subject = _get_header(headers, "Subject")
        sender = _get_header(headers, "From")
        date = _get_header(headers, "Date")
        snippet = full_msg.get("snippet", "")

        output.append(
            f"ID: {msg['id']}\n"
            f"Från: {sender}\n"
            f"Ämne: {subject}\n"
            f"Datum: {date}\n"
            f"Snippet: {snippet}\n"
        )

    return "\n---\n".join(output)


@tool
def read_email(message_id: str) -> str:
    """Läs ett specifikt mejl via dess message_id."""
    service = get_gmail_service()

    full_msg = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )

    headers = full_msg.get("payload", {}).get("headers", [])
    subject = _get_header(headers, "Subject")
    sender = _get_header(headers, "From")
    to = _get_header(headers, "To")
    date = _get_header(headers, "Date")
    snippet = full_msg.get("snippet", "")
    body_text = _extract_plain_text(full_msg.get("payload", {}))

    if not body_text:
        body_text = "[Kunde inte extrahera plaintext-body, visar snippet istället]\n" + snippet

    return (
        f"ID: {message_id}\n"
        f"Från: {sender}\n"
        f"Till: {to}\n"
        f"Ämne: {subject}\n"
        f"Datum: {date}\n\n"
        f"Innehåll:\n{body_text}"
    )