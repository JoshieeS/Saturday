import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from ollama import chat, ChatResponse
from langchain_community.document_loaders import PyPDFLoader

MODEL = "jarvis" 
DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(DIR_PATH / "processed", exist_ok=True)
os.makedirs(DIR_PATH / "wiki", exist_ok=True)
os.makedirs(DIR_PATH / "raw", exist_ok=True)

SCAN_ROOTS: list[Path] = [
    DIR_PATH / "raw",
    # Path("E:/Projects"),   
]

DENY_LIST: set[str] = {
    ".git", "__pycache__", "node_modules", ".obsidian",
    "processed", "wiki",  
    "venv", ".venv", "env",
}

PROCESSED_DIR = DIR_PATH / "processed"   
WIKI_DIR      = DIR_PATH / "wiki"        
LOG_PATH      = DIR_PATH / "wiki" / "log.md"
INDEX_PATH    = DIR_PATH / "wiki" / "index.md"

# Create necessary directories if they don't exist
for d in [PROCESSED_DIR, WIKI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

#Loads Schema currently based on Karpathy's LLM Wiki gist
def load_schema() -> str:
    schema_path = DIR_PATH / "SCHEMA.md"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"SCHEMA.md not found at {schema_path}. "
            "Create it before running the ingester."
        )
    return schema_path.read_text(encoding="utf-8")

#Locates PDFs in the specified scan roots, skipping any directories in the deny list.
def find_pdfs(roots: list[Path], deny: set[str]) -> list[Path]:
    """
    Recursively finds all PDFs under every path in `roots`,
    skipping any directory whose name is in `deny`.
    Returns a sorted, deduplicated list.
    """
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            print(f"[WARN] Scan root does not exist, skipping: {root}")
            continue
        for path in root.rglob("*.pdf"):
            if any(part in deny for part in path.parts):
                continue
            found.add(path.resolve())
    return sorted(found)

#Checks the wiki folder if any pdf has already been digested
def is_already_processed(pdf_path: Path) -> bool:
    """
    A PDF is considered processed if its wiki page exists AND
    the log has a SUCCESS entry for it.
    """
    wiki_target = WIKI_DIR / f"{pdf_path.stem}_wiki.md"
    print(f"  [CHECK] Wiki page: {wiki_target.name} ... ", end="")
    if not wiki_target.exists():
        return False
    if not LOG_PATH.exists():
        return False
    log_text = LOG_PATH.read_text(encoding="utf-8")
    return f"SUCCESS | {pdf_path.name}" in log_text

#Converts PDF to markdown, easy to read by LLM
def pdf_to_markdown(pdf_path: Path) -> Path:
    """
    Converts a PDF to a plain markdown transcript (one section per page).
    Saves to processed/{stem}.md. Returns the path.
    Skips if the transcript already exists.
    """
    out_path = PROCESSED_DIR / f"{pdf_path.stem}.md"
    if out_path.exists():
        print(f"  [TRANSCRIPT] Already exists, reusing: {out_path.name}")
        return out_path

    print(f"  [EXTRACT] Reading {pdf_path.name} ...")
    loader = PyPDFLoader(str(pdf_path), mode="page")
    pages = loader.load()
    total = len(pages)

    sections = []
    for page in pages:
        num     = page.metadata.get("page", 0) + 1
        content = page.page_content.strip()
        sections.append(f"## Page {num}\n\n{content}")

    header = (
        f"# {pdf_path.stem}\n\n"
        f"**Source:** `{pdf_path.name}` | **Pages:** {total}\n\n"
        f"---\n\n"
    )
    full_md = header + "\n\n---\n\n".join(sections)
    out_path.write_text(full_md, encoding="utf-8")
    print(f"  [TRANSCRIPT] {total} pages → {out_path.name}")
    return out_path


# Getting existing wiki pages
def get_existing_wiki_titles() -> list[str]:
    """Returns display names of all existing wiki pages (for cross-linking)."""
    if not WIKI_DIR.exists():
        return []
    return [
        p.stem.replace("_wiki", "").replace("_", " ")
        for p in WIKI_DIR.glob("*_wiki.md")
    ]

#Get existing wiki summaries to help LLM decide on cross-linking
def get_wiki_summaries() -> str:
    """
    Returns a compact summary of existing wiki pages (title + first 3 lines)
    so the LLM can decide what to cross-link intelligently.
    """
    summaries = []
    for p in sorted(WIKI_DIR.glob("*_wiki.md")):
        lines = p.read_text(encoding="utf-8").split("\n")
        # Grab first non-empty lines after the header
        preview = "\n".join(l for l in lines[:10] if l.strip())[:300]
        summaries.append(f"### {p.stem}\n{preview}")
    return "\n\n".join(summaries) if summaries else "No wiki pages exist yet."


#Initialize LLM model
def llm(system: str, user: str, label: str = "") -> str:
    """Single LLM call wrapper with basic retry."""
    if label:
        print(f"  [LLM:{label}] calling {MODEL} ...")
    for attempt in range(3):
        try:
            response: ChatResponse = chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                stream=False,
            )
            return response.message.content.strip()
        except Exception as e:
            print(f"  [WARN] LLM call failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call '{label}' failed after 3 attempts.")


# Pass 1: Getting the LLM to understand the document.
UNDERSTAND_SYSTEM = """
You are an expert knowledge analyst. Your job is to deeply understand documents
and reason about how they fit into a growing knowledge base.

You think carefully before writing anything. You identify:
- What this document is actually about (not just its title)
- The most important facts, claims, and data points
- Every named entity: people, companies, projects, tools, standards, places
- Every concept or method that appears
- How this document relates to or contradicts other documents you know about
- What questions this document answers and what new questions it raises

You are honest about uncertainty. If a page is mostly images or scanned text
with poor OCR, you say so.
"""

def pass1_understand(document_content: str, existing_wiki: str, pdf_name: str) -> str:
    """
    Pass 1: Free-form reasoning. The model thinks out loud about the document.
    """
    user_prompt = f"""
    I need you to deeply understand this document before we write a wiki page for it.
    Think carefully. Don't rush to format anything yet.

    DOCUMENT: {pdf_name}
    ---
    {document_content[:12000]}
    ---

    EXISTING WIKI PAGES (for context and cross-linking):
    {existing_wiki}

    Please reason through the following — take your time:

    1. CORE CONTENT: What is this document actually about? What is its purpose?
    Summarise in 3-5 sentences as if explaining to a colleague.

    2. KEY FACTS & CLAIMS: List the 8-12 most important specific facts, data points,
    findings, or claims. Be precise — include numbers, dates, names where present.

    3. ENTITIES: List every named entity (person, company, project, product, standard,
    regulation, tool, location). For each, note what role they play in the document.

    4. CONCEPTS & METHODS: What engineering, scientific, financial, or technical
    concepts/methods/frameworks appear? List them.

    5. CROSS-LINKS: Looking at the existing wiki pages above, which ones does this
    document genuinely connect to? What is the connection? (Not superficial —
    only list real substantive links.)

    6. CONTRADICTIONS OR TENSIONS: Does anything in this document contradict or
    complicate something in the existing wiki? Be specific.

    7. GAPS & OPEN QUESTIONS: What important questions does this document raise
    but not answer? What would a reader want to know next?

    8. QUALITY ASSESSMENT: Rate the OCR/text quality (if scanned). Note any
    pages that seemed garbled, incomplete, or image-only.

    Output your reasoning as structured prose under each numbered heading.
    This is your thinking — be thorough and honest.
    """
    return llm(UNDERSTAND_SYSTEM, user_prompt, label="UNDERSTAND")


# Pass 2: Creates the wiki based on the understanding from Pass 1, while adhering to the schema.
WRITE_SYSTEM = """
You are a meticulous wiki maintainer writing pages for an Obsidian knowledge base.
You write in clean, structured markdown. You follow the schema exactly.
You use [[wikilinks]] for every entity and concept that has or deserves its own page.
You never make up facts — only what is in the source document.
You link to other pages only when the connection is genuinely meaningful.
"""

def pass2_write(understanding: str, document_content: str,
    schema: str, existing_titles: list[str], pdf_name: str) -> str:
    """
    Pass 2: Write the wiki page, fully informed by the understanding pass.
    Schema is enforced here.
    """
    titles_str = "\n".join(f"- [[{t}]]" for t in existing_titles) or "None yet."

    user_prompt = f"""
    You have already analysed this document. Now write the Obsidian wiki page for it,
    strictly following the schema and using your understanding.

    SCHEMA (follow this exactly):
    ---
    {schema}
    ---

    YOUR UNDERSTANDING OF THE DOCUMENT (use this as your source of truth):
    ---
    {understanding}
    ---

    DOCUMENT TEXT (refer back if you need to verify a specific fact):
    ---
    {document_content[:8000]}
    ---

    EXISTING WIKI PAGES (available for [[wikilinks]]):
    {titles_str}

    RULES YOU MUST FOLLOW:
    1. YAML frontmatter first — title, type, tags, sources, created, updated, confidence.
    confidence = high/medium/low based on text quality from your assessment.
    2. Remove ``````markdown``` tags from top and bottom of the page if you included them in your output.
    3. Every entity you identified gets a [[wikilink]] on first mention.
    Format: [[Entity Name]] — no underscores, natural capitalisation.
    4. Every concept you identified gets a [[wikilink]].
    5. Pick the 5-8 most important terms as [[wikilink]] tags at the bottom
    (these create the Obsidian graph web).
    6. Cross-reference only the existing pages where the connection is substantive.
    7. If you noted contradictions with existing pages, include a ## Contradictions section.
    8. Include a ## Open Questions section with the questions you identified.
    9. Source line at the bottom: **Source:** [[{pdf_name}]]
    10. Do NOT invent facts not present in the document or your understanding.
    11. The page should be genuinely useful — not a template with placeholder text.

    Write the complete wiki page now.
    """
    return llm(WRITE_SYSTEM, user_prompt, label="WRITE")


#Validating the wiki page against the schema, and providing a score and list of issues if any.
VALIDATE_SYSTEM = """
You are a strict schema validator for a markdown wiki. 
You check wiki pages against a schema, the below Rules, and return a JSON report.

RULES YOU MUST FOLLOW:
1. MUST FOLLOW this YAML frontmatter first — title, type, tags, sources, created, updated, confidence. confidence = high/medium/low based on text quality from your assessment.
2. MUST remove ``````markdown``` tags from top and bottom of the page if you included them in your output.
3. Every entity you identified gets a [[wikilink]] on first mention, based on the understanding you provided.
Format: [[Entity Name]] — no underscores, natural capitalisation.
4. Every concept you identified gets a [[wikilink]].
5. Pick the 5-8 most important terms as [[wikilink]] tags at the bottom
(these create the Obsidian graph web).
6. Cross-reference only the existing pages where the connection is substantive.
7. If you noted contradictions with existing pages, include a ## Contradictions section.
8. Include a ## Open Questions section with the questions you identified.
9. Source line at the bottom: **Source:** [[{pdf_name}]]
10. Do NOT invent facts not present in the document or your understanding.
11. The page should be genuinely useful — not a template with placeholder text.
"""

def validate_output(wiki_content: str, schema: str) -> dict:
    """
    Ask the LLM to self-audit the page against the schema.
    Returns a dict: {passed: bool, issues: list[str], score: int}
    """
    user_prompt = f"""
    Check this wiki page against the schema. Return ONLY valid JSON, no markdown.

    SCHEMA:
    {schema[:2000]}

    WIKI PAGE:
    {wiki_content[:3000]}

    Return exactly this JSON structure:
    {{
    "passed": true or false,
    "score": integer 0-100,
    "issues": ["list of specific violations or missing elements"],
    "wikilink_count": integer,
    "has_frontmatter": true or false,
    "has_open_questions": true or false
    }}
    """
    raw = llm(VALIDATE_SYSTEM, user_prompt, label="VALIDATE")
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"passed": False, "score": 0,
                "issues": ["Validator returned non-JSON"], "raw": raw}


# Log.md helper functions: appending entries and updating the index.
def append_log(entry: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def update_index(pdf_name: str, wiki_path: Path, score: int, wikilinks: int):
    """Adds or updates a row in wiki/index.md."""
    row = (f"| [[{wiki_path.stem}]] | `{pdf_name}` | "
           f"{datetime.now().strftime('%Y-%m-%d')} | {score}/100 | {wikilinks} links |\n")

    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(
            "# Wiki Index\n\n"
            "| Page | Source | Date | Score | Links |\n"
            "|------|--------|------|-------|-------|\n",
            encoding="utf-8"
        )

    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(row)


# PDF Ingestion pipeline
def ingest_pdf(pdf_path: Path, schema: str):
    print(f"\n{'='*60}")
    print(f"FILE: {pdf_path.name}")
    print(f"PATH: {pdf_path}")
    print(f"{'='*60}")

    if is_already_processed(pdf_path):
        print(f"  [SKIP] Already in wiki with SUCCESS log entry.")
        return

    transcript_path = pdf_to_markdown(pdf_path)
    document_content = transcript_path.read_text(encoding="utf-8")
    existing_wiki    = get_wiki_summaries()
    existing_titles  = get_existing_wiki_titles()

    print(f"  [PASS 1] Understanding document ...")
    understanding = pass1_understand(document_content, existing_wiki, pdf_path.name)
    print(f"  [PASS 1] Complete ({len(understanding)} chars)")

    print(f"  [PASS 2] Writing wiki page ...")
    wiki_content = pass2_write(
        understanding, document_content, schema, existing_titles, pdf_path.name
    )
    print(f"  [PASS 2] Complete ({len(wiki_content)} chars)")

    print(f"  [VALIDATE] Checking schema compliance ...")
    validation = validate_output(wiki_content, schema)
    score      = validation.get("score", 0)
    issues     = validation.get("issues", [])
    wikilinks  = validation.get("wikilink_count", 0)
    print(f"  [VALIDATE] Score: {score}/100 | Wikilinks: {wikilinks}")
    if issues or score < 70:
        for issue in issues:
            print(f"    ⚠ {issue}")
        print(f"  [VALIDATE] Score {score}/100")   
        print(f"  [REPAIR] requesting schema repair ...")
        repair_prompt = f"""
        The wiki page you wrote scored {score}/100 against the schema.
        Issues found: {chr(10).join(f'- {i}' for i in issues)}

        Here is your current page:
        ---
        {wiki_content}
        ---

        Fix ALL the listed issues while keeping all the good content.
        Output the complete corrected wiki page.
        """
        wiki_content = llm(WRITE_SYSTEM, repair_prompt, label="REPAIR")
        # Re-validate
        validation = validate_output(wiki_content, schema)
        score      = validation.get("score", 0)
        print(f"  [REPAIR] New score: {score}/100")

    #Write wiki page
    wiki_filename = WIKI_DIR / f"{pdf_path.stem}_wiki.md"
    source_header = f"\n\n---\n\n**Source:** [[{pdf_path.name}]]\n"
    wiki_filename.write_text(wiki_content + source_header, encoding="utf-8")
    print(f"  [WRITE] → {wiki_filename.name}")

    # Log
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "SUCCESS" if score >= 70 else "PARTIAL"
    log_entry = (
        f"\n## [{ts}] {status} | {pdf_path.name}\n"
        f"- Wiki page: [[{wiki_filename.stem}]]\n"
        f"- Schema score: {score}/100\n"
        f"- Wikilinks: {wikilinks}\n"
        f"- Issues: {'; '.join(issues) if issues else 'none'}\n"
        f"- Source: {pdf_path}\n"
    )
    append_log(log_entry)
    update_index(pdf_path.name, wiki_filename, score, wikilinks)

    print(f"  [DONE] {status} — schema {score}/100")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════╗")
    print("║        SATURDAY BRAIN INGESTER         ║")
    print("╚══════════════════════════════════════╝\n")

    schema = load_schema()
    print(f"[SCHEMA] Loaded ({len(schema)} chars)")

    pdfs = find_pdfs(SCAN_ROOTS, DENY_LIST)
    print(f"[SCAN] Found {len(pdfs)} PDF(s) across {len(SCAN_ROOTS)} root(s)")
    for p in pdfs:
        print(f"  • {p}")

    if not pdfs:
        print("[DONE] Nothing to process.")
        return

    for pdf in pdfs:
        try:
            ingest_pdf(pdf, schema)
        except Exception as e:
            print(f"\n[ERROR] Failed on {pdf.name}: {e}")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            append_log(f"\n## [{ts}] FAILED | {pdf.name}\n- Error: {e}\n")

    print(f"\n[DONE] Wiki has {len(list(WIKI_DIR.glob('*_wiki.md')))} page(s).")
    print(f"       Open the 'wiki/' folder as your Obsidian vault.")


if __name__ == "__main__":
    main()