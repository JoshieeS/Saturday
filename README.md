# Saturday
My twist of Friday

ingest.py — Schema-strict wiki ingester for Obsidian.
researcher.py - Pulls links created in Wiki for concepts and runs a web search to find articles

Philosophy: the LLM runs in two passes per document.
  Pass 1 — UNDERSTAND: free-form reasoning about the document's content,
            entities, concepts, and how it connects to the existing wiki.
  Pass 2 — WRITE: produce the wiki page, strictly constrained to the schema,
            informed by the understanding pass.

This two-pass approach prevents the model from pattern-matching on the schema
template and filling it in blindly. It has to think first.

## Limitations
Works only for .pdf files

## Project Setup

1. Need to create an obsidian vault in the project directory
2. Drop a pdf into the `raw/` folder
3. Run ingest.py followed by researcher.py and watch the magic happen
