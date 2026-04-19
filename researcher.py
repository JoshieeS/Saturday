import os
import re
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from ollama import Client, chat, web_search, web_fetch
import ingest

MODEL = "Jarvis"
API_KEY = "9690f24a59c846f3bc3c0b6558c64c65.Sa3GxrjD37xl3NOeg57p0V1L"

client = Client(
    host="http://localhost:11434",
    headers={
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
)

def extract_missing_concepts(wiki_dir: Path) -> dict[str, list[str]]:
    """Scans for [[links]] that don't have a corresponding _wiki.md file."""
    link_pattern = re.compile(r'\[\[(.*?)\]\]')
    concept_contexts = defaultdict(list)
    
    for wiki_file in wiki_dir.glob("*_wiki.md"):
        content = wiki_file.read_text(encoding="utf-8")
        for line in content.split("\n"):
            matches = link_pattern.findall(line)
            for match in matches:
                concept = match.split("|")[0].strip()
                if len(concept) > 1 and " " in line:
                    concept_contexts[concept].append(line.strip()[:200])
                
    missing_concepts = {}
    for concept, contexts in concept_contexts.items():
        safe_name = concept.replace(" ", "_").replace("/", "-")
        target_file = wiki_dir / f"{safe_name}_wiki.md"
        if not target_file.exists():
            missing_concepts[concept] = contexts[:3]
    return missing_concepts

def generate_search_query(concept: str, contexts: list[str]) -> str:
    """Uses LLM to craft a high-signal search query based on file context."""
    context_str = "\n".join(f"- {c}" for c in contexts)
    system = "You are a research assistant. Output ONLY a search query string."
    user = f"Context:\n{context_str}\n\nWrite a search query to find research articles/technical details/web articles for: {concept}"
    
    response = chat(model=MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    print(f"    [QUERY] {response.message.content.strip()}")
    return response.message.content.strip().strip('"\'')

UNDERSTAND_SYSTEM="""
You are an expert knowledge analyst. Your job is to deeply understand the research document provided, extract its key insights,
and reason about how they fit into a growing knowledge base.

You think carefully before writing anything. You identify:
- What this website is actually about (not just its title)
- The most important facts, claims, and data points
- Every named entity: people, companies, projects, tools, standards, places
- Every concept or method that appears
- How this website relates to or contradicts other websites you know about
- What questions this website answers and what new questions it raises

You are honest about uncertainty. If a page is mostly images or scanned text
with poor OCR, you say so.
"""

def llm_understand(wiki_dir: Path, concept: str, research_data: list[dict]):
    """
    Pass 1: Free-form reasoning. The model thinks out loud about the document.
    """
    user_prompt = f"""
    I need you to deeply understand this below research content before we write a wiki page for it.
    Think carefully. Don't rush to format anything yet.
    ---
    {concept}
    ---

    EXISTING WIKI PAGES (for context and cross-linking):
    {wiki_dir}

    Web Search
    {research_data}

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
    return ingest.llm(UNDERSTAND_SYSTEM, user_prompt, label="UNDERSTAND")


def write_concept_file(concept: str, research_data: list[dict]):
    # Load schema and prepare sources
    schema = ingest.load_schema()
    sources = ", ".join(entry.get('url','') for entry in research_data if entry.get('url'))

    user_prompt = f"""
    Based on the research below for the concept "{concept}", write an Obsidian wiki page following the schema exactly.

    SCHEMA:
    ---
    {schema}
    ---

    RESEARCH (use this as your source of truth):
    ---
    {research_data}
    ---

    RULES YOU MUST FOLLOW:
    1. YAML frontmatter first — title, type, tags, sources, created, updated, confidence.
    confidence = high/medium/low based on text quality from your assessment.
    2. Remove code fences and ```markdown``` tags if present.
    3. Every entity you identified gets a [[wikilink]] on first mention.
    Format: [[Entity Name]] — no underscores, natural capitalisation.
    4. Every concept you identified gets a [[wikilink]].
    5. Pick the 5-8 most important terms as [[wikilink]] tags at the bottom.
    6. Cross-reference only existing pages where the connection is substantive.
    7. If you noted contradictions with existing pages, include a ## Contradictions section.
    8. Include a ## Open Questions section with the questions you identified.
    9. Source line at the bottom: **Sources:** {sources}
    10. Do NOT invent facts not present in the sources.
    11. The page should be genuinely useful — not a template with placeholder text.

    Write the complete wiki page now (no surrounding markdown fences).
    """

    return ingest.llm(ingest.WRITE_SYSTEM, user_prompt, label=f"WRITE_CONCEPT:{concept}")

def validate_concept(concept_content: str, schema: str):
    user_prompt = f"""
    Check this wiki page against the schema. Return ONLY valid JSON, no markdown.

    SCHEMA:
    {schema[:2000]}

    WIKI PAGE:
    {concept_content}

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
    raw = ingest.llm(ingest.VALIDATE_SYSTEM, user_prompt, label="VALIDATE")
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"passed": False, "score": 0,
                "issues": ["Validator returned non-JSON"], "raw": raw}
    

def research_missing_links(wiki_dir: Path):
    """Discover missing concept pages, research them, and run the full ingest-style pipeline
    (UNDERSTAND -> WRITE -> VALIDATE -> REPAIR -> WRITE FILE -> LOG/INDEX) for each concept.
    """
    print("\n--- Starting Autonomous Research ---")

    # Ensure schema and directories are available (reuse ingest setup)
    schema = ingest.load_schema()
    print(f"[SCHEMA] Loaded ({len(schema)} chars)")
    
    missing = extract_missing_concepts(wiki_dir)

    for concept, contexts in missing.items():
        print(f"\n[CONCEPT] {concept}")

        # 1. Generate Query
        query = generate_search_query(concept, contexts)

        # 2. Search
        try:
            search_response = client.web_search(query=query)
            urls = [res.get('url') for res in search_response.get('results', []) if res.get('url')]
            print(urls)
        except Exception as e:
            print(f"    [ERR] Search failed: {e}")
            continue

        # 3. Fetch top results
        research_results = []
        for url in urls[:3]:
            try:
                fetch_res = client.web_fetch(url=url)
                content = getattr(fetch_res, 'content', fetch_res.get('content', '')) if fetch_res else ''
                research_results.append({"url": url, "content": content})
                print(f"[FETCHED] {url}, {len(content)} chars")
            except Exception as e:
                print(f"[ERR] Fetch failed for {url}: {e}")

        if not research_results:
            continue

        # Compile fetched content into a single document string
        compiled_text = ""
        for entry in research_results:
            compiled_text += f"\n\nSource: {entry['url']}\nContent:\n{entry['content']}\n"

        #4. Understand with LLM
        understanding = llm_understand(wiki_dir, concept, compiled_text)

        #5. Drafting wiki knowledge
        concept_content = write_concept_file(concept, understanding)

        #6. Validate output
        validation = validate_concept(concept_content, schema)

        score = validation.get("score", 0)
        issues = validation.get("issues", [])
        wikilinks = validation.get("wikilink_count", 0)
        print(f"[VALIDATE] Score: {score}/100 | Wikilinks: {wikilinks}")
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
            {concept_content}
            ---

            Fix ALL the listed issues while keeping all the good content.
            Output the complete corrected wiki page.
            """
            concept_content = ingest.llm(ingest.WRITE_SYSTEM, repair_prompt, label="REPAIR")
            # Re-validate
            validation = validate_concept(concept_content, schema)
            score = validation.get("score", 0)
            print(f"  [REPAIR] New score: {score}/100")

    #Write wiki page
    wiki_filename = WIKI_DIR / f"{concept.stem}_wiki.md"
    source_header = f"\n\n---\n\n**Source:** [[{concept.name}]]\n"
    wiki_filename.write_text(concept_content + source_header, encoding="utf-8")
    print(f"  [WRITE] → {wiki_filename.name}")
        
    # Log
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "SUCCESS" if score >= 70 else "PARTIAL"
    log_entry = (
        f"\n## [{ts}] {status} | {concept.name}\n"
        f"- Wiki page: [[{wiki_filename.stem}]]\n"
        f"- Schema score: {score}/100\n"
        f"- Wikilinks: {wikilinks}\n"
        f"- Issues: {'; '.join(issues) if issues else 'none'}\n"
        f"- Source: {concept}\n"
    )
    ingest.append_log(log_entry)
    ingest.update_index(concept.name, wiki_filename, score, wikilinks)
    print(f"  [DONE] {status} — schema {score}/100")
    
if __name__ == "__main__":
    DIR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
    WIKI_DIR = ingest.WIKI_DIR
    research_missing_links(WIKI_DIR)
