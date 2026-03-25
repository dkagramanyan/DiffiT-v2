#!/bin/bash
# Evaluate FID-50K and other metrics for DiffiT generated samples.
#
# Usage:
#   bash eval_run.sh                              # defaults (256)
#   bash eval_run.sh 512 ./samples/512            # custom resolution and path

RESOLUTION=${1:-256}
SAMPLE_DIR=${2:-./samples/${RESOLUTION}}
REF_BATCH=${3:-./VIRTUAL_imagenet${RESOLUTION}_labeled.npz}

# Find the sample .npz file
SAMPLE_BATCH=$(ls ${SAMPLE_DIR}/samples_*.npz 2>/dev/null | head -1)

if [ -z "$SAMPLE_BATCH" ]; then
    echo "Error: No samples .npz found in ${SAMPLE_DIR}"
    echo "Run sample.py first to generate samples."
    exit 1
fi

echo "Reference: ${REF_BATCH}"
echo "Samples:   ${SAMPLE_BATCH}"
echo ""

python evaluator.py --ref-batch "${REF_BATCH}" --sample-batch "${SAMPLE_BATCH}"
