#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/$ENV_NAME
pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install -e $REPO_DIR
exec "$@"