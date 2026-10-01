#!/bin/bash
set -e

KRYPTEX_USER="${KRYPTEX_USER:-krxX7DNVJE}"
POOL="${POOL:-prl.kryptex.network:7048}"

WORKER="salad-$(hostname | cut -c1-8)"

echo "======================================"
echo " Pearl miner iniciando"
echo " Worker: $WORKER"
echo " Pool: $POOL"
echo "======================================"

exec /miner/SRBMiner-MULTI \
    --disable-cpu \
    --algorithm pearlhash \
    --pool "$POOL" \
    --wallet "${KRYPTEX_USER}.${WORKER}"
