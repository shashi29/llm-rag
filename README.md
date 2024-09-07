# llm-rag

uv init llm-rag
uv add pdfminer.six docx2txt tqdm

uv pip install pdfminer.six docx2txt tqdm

uv run pdfminer.six docx2txt check

source .venv/bin/activate

uv pip sync docs/requirements.txt

sudo apt-get install antiword