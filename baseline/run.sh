#!/bin/bash
conda activate mirgeDriver.Y1flatPlate
mpirun -n 1 python -u -O -m mpi4py plate.py
