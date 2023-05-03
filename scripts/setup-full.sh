#!/bin/bash

./scripts/download_dataset.sh flag_simple data
./scripts/convert_dataset flag_simple train &
./scripts/convert_dataset flag_simple valid &


./scripts/download_dataset.sh deforming_plate data
./scripts/convert_dataset deforming_plate train &
./scripts/convert_dataset deforming_plate valid &

./scripts/download_dataset.sh cylinder_flow data
./scripts/convert_dataset cylinder_flow train &
./scripts/convert_dataset cylinder_flow valid &

# ./scripts/download_dataset.sh airfoil data
# ./scripts/convert_dataset airfoil train &
# ./scripts/convert_dataset airfoil valid &

wait
