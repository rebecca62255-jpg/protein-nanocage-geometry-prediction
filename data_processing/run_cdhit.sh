#!/bin/bash

# Cluster highly similar protein sequences using CD-HIT.
#
# Input:
#   data/sequences.fasta
#
# Outputs:
#   data/sequences_nr.fasta
#   data/sequences_nr.fasta.clstr
#
# Sequences sharing >=90% sequence identity are clustered together.
# One representative sequence is retained for each cluster.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

INPUT="$ROOT/data/sequences.fasta"
OUTPUT="$ROOT/data/sequences_nr.fasta"

echo "Running CD-HIT at 90% sequence identity..."
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

cd-hit \
    -i "$INPUT" \
    -o "$OUTPUT" \
    -c 0.90 \
    -n 5 \
    -d 0 \
    -M 0 \
    -T 0

echo
echo "CD-HIT finished."

echo -n "Input sequences: "
grep -c "^>" "$INPUT"

echo -n "Representative sequences: "
grep -c "^>" "$OUTPUT"

echo -n "Clusters: "
grep -c "^>Cluster" "$OUTPUT.clstr"

echo
echo "Cluster file:"
echo "$OUTPUT.clstr"
