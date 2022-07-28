# pyspark Template

The purpose of this container is to create a container with Pyspark - which is the Python version of Apache Spark. Pyspark is the analysis tool for parquet/csv/json etc. file formats.


Files:
- `Dockerfile` to build image
- `environment.yml` to create virtual conda environment with the desired packages
- `test.py` is a basic python script to test if pyspark installed and ran properly.

Installing `java` when using `miniconda` as base is a bit tricky. This container handles all the issues with related to `Java`, `JAVA_PATH` and `pyspark`


The image can be used by:
`docker pull quay.io/lifebitaiorg/pyspark`

### Before testing

```
git clone https://github.com/lifebit-ai/Docker-containers
cd Docker-containers
git checkout adds-pyspark-image
```

### How to test

Tool can be tested both:
1. `docker run -it quay.io/lifebitaiorg/pyspark:latest` for automated testing,
2. `docker run -it --entrypoint /bin/bash quay.io/lifebitaiorg/pyspark:latest` for interactive testing in bash:
```
root#/: conda run -n pyspark-venv python test.py
```

test script will basically read a parquet file and print its results.