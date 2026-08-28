import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from liteparse import LiteParse

from app.core.config import get_settings

settings = get_settings()

PDF_PATH = "Data/kech1a1.pdf"  # Update path if needed
OUTPUT_MD_PATH = "parsed_output.md"


def main():
    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        print(f"❌ File not found: {PDF_PATH}")
        sys.exit(1)

    print(f"📄 Parsing: {pdf_file.name}")
    parser = LiteParse(output_format="markdown",   # "json" | "text" | "markdown"
                image_mode="placeholder",   # "placeholder" | "off" | "embed"
                extract_links=True,         # render [text](url) link syntax (default: True)
                dpi=250,)
    parse_result = parser.parse(str(pdf_file))

    # Extract Markdown content
    if hasattr(parse_result, "to_markdown"):
        markdown_content = parse_result.to_markdown()
    elif hasattr(parse_result, "markdown"):
        markdown_content = parse_result.markdown
    elif hasattr(parse_result, "text"):
        markdown_content = parse_result.text
    else:
        markdown_content = str(parse_result)

    # Save to Markdown file
    output_path = Path(OUTPUT_MD_PATH)
    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"💾 Saved full markdown output to: {output_path.resolve()}")

    # Diagnostics
    raw_len = len(markdown_content)
    raw_lines = len(markdown_content.splitlines())
    print("\n--- 📊 Parser Diagnostics ---")
    print(f"Total raw characters : {raw_len:,}")
    print(f"Total raw lines      : {raw_lines:,}")
    print(f"Sample (first 300 chars):\n{repr(markdown_content[:300])}\n")

    # Run Text Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_text(markdown_content)
    print("--- ✂️ Splitter Diagnostics ---")
    print(f"Configured Chunk Size   : {settings.CHUNK_SIZE}")
    print(f"Configured Chunk Overlap: {settings.CHUNK_OVERLAP}")
    print(f"Total Chunks Generated  : {len(chunks)}")

    if chunks:
        chunk_lens = [len(c) for c in chunks]
        avg_len = sum(chunk_lens) / len(chunk_lens)
        min_len = min(chunk_lens)
        max_len = max(chunk_lens)

        print(f"Min Chunk Length        : {min_len} chars")
        print(f"Max Chunk Length        : {max_len} chars")
        print(f"Avg Chunk Length        : {avg_len:.1f} chars")

        print("\n--- 📝 Sample First Chunk ---")
        print(chunks[0])
        print("\n--- 📝 Sample Last Chunk ---")
        print(chunks[-1])


if __name__ == "__main__":
    main()