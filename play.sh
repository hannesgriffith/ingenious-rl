#!/bin/bash

cd code

echo Good luck!
NUMBA_DISABLE_JIT=1 python -m main settings_play.json

cd ..