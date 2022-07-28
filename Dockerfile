FROM continuumio/miniconda3:4.9.2

LABEL name="quay.io/lifebitaiorg/pyspark" \ 
      description="Docker image containing pyspark and its dependencies for analysis" \
      version="1.0.0" \
      maintainer="Erdal Genc <erdal@lifebit.ai>"

## Installing required packages for Ubuntu
RUN apt-get --allow-releaseinfo-change update \
    && apt-get -qq update -y \
    && apt-get install -y iputils-ping \
    && apt-get -qq install -y --no-install-recommends \
    build-essential git make clang libboost-dev postgresql-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*

## Installing required packages for Java
RUN apt-get --allow-releaseinfo-change update \
    && apt-get update -y \
    && apt-get -y upgrade \
    && apt-get install -y build-essential \
    && apt-get install -y software-properties-common \
    && apt-get -y install gnupg jq \
    && wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add - \
    && apt-get -y update \
    && apt-get -y upgrade \
    && add-apt-repository --yes https://adoptopenjdk.jfrog.io/adoptopenjdk/deb/ \
    && apt-get install --fix-broken \
    && apt-get update -y && apt-get install -y adoptopenjdk-8-hotspot && apt-get install wget -y \
    && apt-get install unzip -y && apt-get install zip -y \
    && wget -N https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/en/sentiment.parquet.zip -P . \
    && unzip sentiment.parquet.zip

## Defining Java path, configuring the version and checking if everything goes weel
ENV JAVA_HOME /usr/lib/jvm/adoptopenjdk-8-hotspot-amd64/
RUN export JAVA_HOME

RUN update-java-alternatives --set adoptopenjdk-8-hotspot-amd64
RUN java -version

## Creating Conda environment
ARG ENV_NAME="pyspark-venv"

RUN conda install -c conda-forge mamba
ENV PATH /opt/conda/envs/${ENV_NAME}/bin:$PATH

## Installing Conda environment
COPY environment.yml .
RUN mamba env create --quiet --name ${ENV_NAME} --file /environment.yml && conda clean -a
COPY test.py .

# Dump the details of the installed packages to a file for posterity
RUN mamba env export --name ${ENV_NAME} > ${ENV_NAME}_exported.yml

CMD ["/bin/bash"]