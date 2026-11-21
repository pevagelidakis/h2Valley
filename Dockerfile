FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Non-root user (uid 1000 is also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 app \
 && mkdir /home/app/h2valley && chown app:app /home/app/h2valley
WORKDIR /home/app/h2valley

COPY requirements.txt .
RUN pip install -r requirements.txt

# .cache/ is NOT shipped (it's in .dockerignore): cache signatures include each
# source file's absolute path + mtime, so a host-built cache never matches in
# the container. Instead docker-compose mounts a named volume at .cache/, so
# the first start computes everything once and later restarts/rebuilds reuse it.
COPY --chown=app:app . .
# Create .cache owned by `app` so the named volume inherits writable ownership.
RUN mkdir -p .cache && chown app:app .cache
USER app

# One BLAS/OpenMP thread per Python thread. The app runs 4-5 precompute
# threads at once on small matrices; letting each spawn a full-core BLAS pool
# oversubscribes the CPU (~60 threads on 14 cores) and makes startup 3-4x
# slower. PyTorch (GENDA) sets its own thread count via torch.set_num_threads.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# 7860 is the Hugging Face Spaces default; other hosts can override via $PORT.
ENV PORT=7860
EXPOSE 7860

# One worker on purpose: the app holds large in-memory state (SARIMA tensor,
# caches) at module level, so extra workers would each duplicate it. Threads
# serve concurrent requests. No --preload: it would break the daemon
# precompute threads if the cache were ever cold.
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 1800 --access-logfile -
