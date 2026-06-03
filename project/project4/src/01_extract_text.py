import os
import fitz

CORPUS_DIR = "corpus"
OUT_DIR = "corpus_raw"
os.makedirs(OUT_DIR, exist_ok=True)

docs_info = {
    "ONC-HOMICIDIO.pdf": "ONC - Reporte Homicidio",
    "ENVIPE_2025.pdf": "INEGI - ENVIPE 2025",
    "ONC_Anual_2025.pdf": "ONC - Reporte Anual 2025",
}

def table_to_text(table):
    rows = []
    for row in table.extract():
        cleaned = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)

for fname, label in docs_info.items():
    path = os.path.join(CORPUS_DIR, fname)
    outpath = os.path.join(OUT_DIR, fname.replace(".pdf", ".txt"))

    doc = fitz.open(path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()

        ft = page.find_tables()
        tables_text = ""
        for t in ft.tables:
            tables_text += "[TABLA]\n" + table_to_text(t) + "\n[/TABLA]\n"

        if tables_text:
            page_text = text + "\n\n" + tables_text
        else:
            page_text = text

        pages.append(page_text)

    full = "\n\n--- PAGE BREAK ---\n\n".join(pages)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(full)

    print(f"{label}: {len(pages)} paginas, {len(full)} caracteres")
    doc.close()
