# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-scikit-learn_scikit-learn-testbed:0.22

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 7e7b5092991cf7a7cf6bd95d56b08deef5eb9847

RUN pip install -v --no-build-isolation -e .

WORKDIR /opt