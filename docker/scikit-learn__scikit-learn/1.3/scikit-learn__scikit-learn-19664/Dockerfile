# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-scikit-learn_scikit-learn-testbed:1.3

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 2620a5545a806ee416d9d10e07c2de30cdd9bf20

RUN pip3 install Cython==0.29.37 numpy==1.23.5

RUN pip install -v --no-use-pep517 --no-build-isolation -e .

WORKDIR /opt

ENTRYPOINT [ "/bin/bash" ]