# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-scikit-learn_scikit-learn-testbed:0.22

WORKDIR /opt/scikit-learn__scikit-learn

RUN git -c advice.detachedHead=false checkout 6ab8c86c383dd847a1be7103ad115f174fe23ffd

RUN pip install -v --no-build-isolation -e .

WORKDIR /opt