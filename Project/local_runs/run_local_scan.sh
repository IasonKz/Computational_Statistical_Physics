#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build results/raw results/processed figures logs/local parameters
cd build
cmake ..
make -j
cd ..

cat > parameters/local_scan_D1_powers2.csv <<'EOF'
run_id,L,T,J,D,Bx,By,Bz,seed,therm,samples,skip,proposal,init
0,8,0.8,1.0,1.0,0.0,0.0,0.0,1,120,180,2,0.7,helical_z
1,8,1.2,1.0,1.0,0.0,0.0,0.0,1,120,180,2,0.7,helical_z
2,8,1.6,1.0,1.0,0.0,0.0,0.0,1,120,180,2,0.7,helical_z
3,8,2.0,1.0,1.0,0.0,0.0,0.0,1,120,180,2,0.7,helical_z
4,16,0.8,1.0,1.0,0.0,0.0,0.0,1,80,100,2,0.7,helical_z
5,16,1.4,1.0,1.0,0.0,0.0,0.0,1,80,100,2,0.7,helical_z
6,16,2.0,1.0,1.0,0.0,0.0,0.0,1,80,100,2,0.7,helical_z
EOF

while IFS=, read -r run_id L T J D Bx By Bz seed therm samples skip proposal init; do
    if [ "$run_id" = "run_id" ]; then continue; fi
    init=$(echo "$init" | tr -d '\r')
    ./build/heisenberg_dmi \
      --L "$L" --T "$T" --J "$J" --D "$D" \
      --Bx "$Bx" --By "$By" --Bz "$Bz" \
      --seed "$seed" --therm "$therm" --samples "$samples" --skip "$skip" \
      --proposal "$proposal" --init "$init" --out results/raw
done < parameters/local_scan_D1_powers2.csv

python3 scripts/collect_results.py
python3 scripts/plot_summary.py
echo "Local scan complete."
