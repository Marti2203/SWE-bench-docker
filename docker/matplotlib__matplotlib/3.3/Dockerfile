# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-matplotlib_matplotlib:bookworm-slim

RUN git -c advice.detachedHead=false checkout 28289122be81e0bc0a6ee0c4c5b7343a46ce2e4e

RUN conda create -n matplotlib__matplotlib__3.3 python=3.8 -y
SHELL ["conda", "run", "-n", "matplotlib__matplotlib__3.3", "/bin/bash", "-c"]

COPY ./docker/matplotlib__matplotlib/3.3/requirements.txt .

RUN pip install -r requirements.txt


COPY --chown=swe-bench:swe-bench qhull-2020-src-8.0.2.tgz /home/swe-bench/

RUN mkdir -p /home/swe-bench/.cache/matplotlib

RUN cp /home/swe-bench/qhull-2020-src-8.0.2.tgz /home/swe-bench/.cache/matplotlib/

RUN cp /home/swe-bench/qhull-2020-src-8.0.2.tgz /home/swe-bench/.cache/matplotlib/b5c2d7eb833278881b952c8a52d20179eab87766b00b865000469a45c1838b7e

COPY qhull-2020-src-8.0.2.tgz /home/swe-bench/matplotlib__matplotlib/build/

USER root

RUN mkdir -p /root/.cache/matplotlib

COPY --chown=root:root qhull-2020-src-8.0.2.tgz /root/.cache/matplotlib/


COPY --chown=root:root qhull-2020-src-8.0.2.tgz /root/.cache/matplotlib/b5c2d7eb833278881b952c8a52d20179eab87766b00b865000469a45c1838b7e

RUN python -m pip install -e .

USER swe-bench


RUN git checkout 28289122be81e0bc0a6ee0c4c5b7343a46ce2e4e .

ENV TESTBED_NAME=matplotlib__matplotlib__3.3
ENV IMAGE_TYPE=conda

WORKDIR /home/swe-bench

COPY swebench_docker swebench_docker

ENTRYPOINT ["./entrypoint.sh"]