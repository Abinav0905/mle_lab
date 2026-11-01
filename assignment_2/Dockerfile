# Apache Airflow 2.10.5 (Debian bookworm, Python 3.11)
FROM apache/airflow:2.10.5-python3.11

# Switch to root to install Java, which PySpark needs
USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # point JAVA_HOME at the installed Java, on both Intel (amd64) and Apple Silicon (arm64) machines
    ln -sfn "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /usr/lib/jvm/java-home

ENV JAVA_HOME=/usr/lib/jvm/java-home

# Switch back to the airflow user before installing Python dependencies
USER airflow

COPY requirements.txt /requirements.txt

# keep Airflow pinned to the image's version while installing our libraries
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /requirements.txt
