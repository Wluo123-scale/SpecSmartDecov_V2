# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
# fastdecov for any desktop with Docker (Linux, macOS, Windows): no compiler or Python setup on the host.
#   docker buildx build --platform linux/amd64,linux/arm64 -t fastdecov .
#   docker run --rm -v "$PWD":/data fastdecov fit spectra out --figures
# The solvers are compiled here once per instruction-set level; at start-up fastdecov.runtime picks the best level
# the CPU supports and checks it on a synthetic map before running.
ARG BASE=ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

FROM ${BASE} AS solvers
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends g++ libgomp1 python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
COPY fastdecov /src/fastdecov
# Same flags as a local build (fastdecov.hw.compile_flags), with -march fixed per level instead of native.
RUN set -e; \
    case "$TARGETARCH" in \
      amd64) levels="x86-64-v4 x86-64-v3 x86-64-v2" ;; \
      arm64) levels="armv8.2-a armv8-a" ;; \
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    for l in $levels; do \
      FASTDECOV_BUILD_DIR=/solvers/$l FASTDECOV_MARCH=$l PYTHONPATH=/src \
        python3 -c "from fastdecov import lm_native; print(lm_native.build_all(force=True))"; \
    done

FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python3 -m venv $VIRTUAL_ENV \
    && pip install --index-url https://pypi.org/simple "numpy==2.5.4" "scipy==1.18.1" "matplotlib==3.11.2"
COPY pyproject.toml README.md LICENSE /tmp/src/
COPY fastdecov /tmp/src/fastdecov
RUN pip install --index-url https://pypi.org/simple --no-deps /tmp/src && rm -rf /tmp/src
COPY --from=solvers /solvers /opt/fastdecov/solvers
RUN chmod -R a-w /opt/fastdecov/solvers && useradd --create-home --uid 10001 fastdecov
USER fastdecov
WORKDIR /data
ENV FASTDECOV_SOLVERS=/opt/fastdecov/solvers MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
ENTRYPOINT ["python3", "-m", "fastdecov.runtime", "exec", "--"]
CMD ["info"]
