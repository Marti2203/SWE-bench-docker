# Generated by generate_dockerfiles.py

FROM autocoderover/swe-bench-astropy_astropy:bookworm-slim



COPY --from=autocoderover/swe-bench-pyenvs:bookworm-slim /opt/pyenv/versions/3.9.19 /opt/pyenv/versions/3.9.19

RUN pyenv rehash && \
    pyenv global 3.9.19 && \
    pyenv versions

WORKDIR /opt/astropy__astropy

RUN git -c advice.detachedHead=false checkout 362f6df12abf9bd769d4915fabf955c993ea22cf

RUN pip install setuptools==68.0.0

RUN pip install attrs==23.1.0 exceptiongroup==1.1.3 execnet==2.0.2 hypothesis==6.82.6 iniconfig==2.0.0 numpy==1.23.4 packaging==23.1 pluggy==1.3.0 psutil==5.9.5 pyerfa==2.0.0.3 pytest-arraydiff==0.5.0 pytest-astropy-header==0.2.2 pytest-astropy==0.10.0 pytest-cov==4.1.0 pytest-doctestplus==1.0.0 pytest-filter-subpackage==0.1.2 pytest-mock==3.11.1 pytest-openfiles==0.5.0 pytest-remotedata==0.4.0 pytest-xdist==3.3.1 pytest==7.4.0 PyYAML==6.0.1 sortedcontainers==2.4.0 tomli==2.0.1

RUN sed -i 's/^from setuptools.dep_util import newer_group/import setuptools; from setuptools import modified; newer_group = modified.newer_group /g' ./astropy/wcs/setup_package.py


RUN pip install -e .[test]


RUN git checkout 362f6df12abf9bd769d4915fabf955c993ea22cf .

ENV TESTBED_NAME=astropy__astropy__5.2
ENV IMAGE_TYPE=pyenv

WORKDIR /opt

COPY swebench_docker swebench_docker
COPY docker/pyenv/entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]