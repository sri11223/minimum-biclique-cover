#!/bin/bash
set -euo pipefail
install -d -m 755 /root/biclique_cover_submission
install -m 755 /solution/biclique_cover_reference.py /root/biclique_cover_submission/solver.py
install -d -m 755 /root/oracle_check
python3 /root/biclique_cover_submission/solver.py /root/data/instances.json /root/oracle_check
