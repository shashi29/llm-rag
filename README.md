# llm-rag

uv init llm-rag
uv add pdfminer.six docx2txt

uv run pdfminer.six docx2txt check

source .venv/bin/activate

uv pip sync docs/requirements.txt

sudo apt-get install antiword