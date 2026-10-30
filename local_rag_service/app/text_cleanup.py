import re


def repair_pdf_encoding(text: str) -> str:
    """Repair the CP1251-as-Latin1 Russian text layer found in some lecture PDFs.

    Require substantial mojibake and no existing Russian text; leave other PDFs alone.
    Mathematical symbols outside the affected character range are preserved.
    """
    runs = re.findall(r"[À-ÿ]+", text)
    if (
        sum(map(len, runs)) < 40
        or sum(len(run) >= 3 for run in runs) < 4
        or re.search(r"[А-Яа-яЁё]", text)
    ):
        return text
    return re.sub(r"[À-ÿ]+", lambda m: m[0].encode("latin1").decode("cp1251"), text)
