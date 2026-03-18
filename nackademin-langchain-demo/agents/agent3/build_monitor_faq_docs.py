from pathlib import Path
import re
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "documents" / "monitor_faq"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://help.monitorerp.com/SE-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Home.htm",
    "https://help.monitorerp.com/se-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Stock/FAQ_Stock_Home.htm",
    "https://help.monitorerp.com/SE-MONITOR_G5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Miscellaneous/FAQ_Technic_Installation.htm",
    "https://help.monitorerp.com/se-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Sales/FAQ_CustomerOrder.htm",
    "https://help.monitorerp.com/SE-MONITOR_G5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Finance/FAQ_Invoices.htm",
    "https://help.monitorerp.com/SE-MONITOR_G5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Sales/FAQ_Customers.htm",
    "https://help.monitorerp.com/se-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Finance/FAQ_Payments.htm",
    "https://help.monitorerp.com/SE-MONITOR_G5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Miscellaneous/FAQ_UserRights.htm",
    "https://help.monitorerp.com/se-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Stock/FAQ_CaseManagement.htm",
    "https://help.monitorerp.com/SE-MONITOR_G5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_EDIShipping/FAQ_EDIShipping.htm",
    "https://help.monitorerp.com/se-monitor_g5/sv-se/Content/Topics/UserGuide/FAQ/FAQ_Stock/FAQ_StockCount.htm",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9åäö]+", "_", text)
    return text.strip("_")


def clean_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    # Försök hitta titel
    title_tag = soup.find(["h1", "h2", "h3"])
    title = title_tag.get_text(" ", strip=True) if title_tag else "monitor_faq"

    # Ta bort script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Enkel rensning av brus
    filtered = []
    skip_words = {
        "hoppa över till huvudinnehåll",
        "konto",
        "inställningar",
        "logga ut",
        "placeholder",
        "filter:",
        "alla filer",
        "skicka sökning",
    }

    for line in lines:
        if line.lower() in skip_words:
            continue
        if line == "Image":
            continue
        filtered.append(line)

    cleaned = "\n".join(filtered)
    return title, cleaned


def main():
    for url in URLS:
        print(f"Hämtar: {url}")
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        title, cleaned_text = clean_text(response.text)
        filename = slugify(title) + ".txt"

        output_path = OUTPUT_DIR / filename
        content = f"KÄLLA: {url}\nTITEL: {title}\n\n{cleaned_text}\n"

        output_path.write_text(content, encoding="utf-8")
        print(f"Sparade: {output_path}")


if __name__ == "__main__":
    main()