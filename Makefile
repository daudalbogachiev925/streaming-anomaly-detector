.PHONY: install test stream serve

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

stream:
	python scripts/run_stream.py

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
