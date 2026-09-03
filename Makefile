.PHONY: data pipeline serve test all clean
data:      ; bash get_data.sh
pipeline:  ; python pipeline.py && python oos.py && python analytics.py && \
             python depth.py && python finalize_metrics.py && python export_web.py
serve:     ; uvicorn serve:app --port 8000
test:      ; python -m pytest tests/ -q
all: pipeline test
clean:     ; rm -rf artifacts/*.parquet artifacts/*.csv artifacts/*.json web/data.json
