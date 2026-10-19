FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Non-root user (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 app \
 && mkdir /home/app/h2valley && chown app:app /home/app/h2valley
WORKDIR /home/app/h2valley

COPY requirements.txt .
RUN pip install -r requirements.txt

# .cache/ is shipped with the image (see cache_utils.py: entries are keyed on
# the CSV's content hash, so they stay valid). Regenerate it locally with
# `python warm_cache.py` (~1 h, dominated by the SARIMA fits) whenever spg.csv
# or an analysis setting changes. Without it the app still works, but every
# cold start recomputes in the background and the tabs fill in progressively.
COPY --chown=app:app . .
USER app

# 7860 is the Hugging Face Spaces default; other hosts can override via $PORT.
ENV PORT=7860
EXPOSE 7860

# One worker on purpose: the app holds large in-memory state (SARIMA tensor,
# caches) at module level, so extra workers would each duplicate it. Threads
# serve concurrent requests. No --preload: it would break the daemon
# precompute threads if the cache were ever cold.
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600 --access-logfile -
