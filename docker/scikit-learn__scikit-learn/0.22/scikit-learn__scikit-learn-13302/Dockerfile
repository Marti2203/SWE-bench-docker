# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-scikit-learn_scikit-learn-testbed:0.22

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 4de404d46d24805ff48ad255ec3169a5155986f0

RUN pip install -v --no-build-isolation -e .

WORKDIR /opt