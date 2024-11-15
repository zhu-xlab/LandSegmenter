#!/bin/bash
pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install -e $REPO_DIR
exec "$@"