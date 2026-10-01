FROM nvidia/cuda:12.8.2-base-ubuntu24.04

RUN apt-get update && \
    apt-get install -y wget ca-certificates tar && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /miner

RUN wget -q \
    https://github.com/doktor83/SRBMiner-Multi/releases/download/3.7.0/SRBMiner-Multi-3-7-0-Linux.tar.gz \
    && tar -xzf SRBMiner-Multi-3-7-0-Linux.tar.gz \
    && cp SRBMiner-Multi-3-7-0/SRBMiner-MULTI /miner/SRBMiner-MULTI \
    && chmod +x /miner/SRBMiner-MULTI \
    && rm -rf SRBMiner-Multi-3-7-0*

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
