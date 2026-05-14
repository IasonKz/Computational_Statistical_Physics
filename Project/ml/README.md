# Machine Learning Pipeline

The mandatory ML part has three approaches:

1. **Feature baseline**  
   Uses physics observables from `results/processed/summary.csv`.
   This is a sanity check and interpretable baseline.

2. **Structure-factor model**  
   Reads saved configurations, computes `S(q)` features, and trains a classifier.
   This is physics-informed but still configuration-based.

3. **3D CNN**  
   This is the **main model**. It consumes raw spin tensors with shape:

   ```text
   (N, 3, L, L, L)
   ```

   where channels are `Sx, Sy, Sz`.

## Recommended ML data collection

```text
L = 8, 16, 32
20 training temperatures + 10 near-transition temperatures
5 seeds
50 saved configurations per (L,T,seed)
Total: 3 * 30 * 5 * 50 = 22500 configurations
```

Because raw tensors have different sizes for different `L`, build and train one model per lattice size:

```text
ml_data_L8
ml_data_L16
ml_data_L32
```

## 1. Build one dataset per L

```bash
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L8  --L 8  --low-max 0.95 --high-min 1.80 --store-cnn
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L16 --L 16 --low-max 0.95 --high-min 1.80 --store-cnn
python3 ml/build_dataset.py --config-dir results/configs_ml --out-dir ml_data_L32 --L 32 --low-max 0.95 --high-min 1.80 --store-cnn
```

## 2. Feature baseline

```bash
python3 ml/train_feature_baseline.py --summary results/processed/summary.csv --low-max 0.95 --high-min 1.80
```

## 3. Structure-factor model

```bash
python3 ml/build_structure_factor_dataset.py --config-dir results/configs_ml --out-dir ml_data_sf_L16 --L 16 --nmax 8 --low-max 0.95 --high-min 1.80
python3 ml/train_structure_factor_model.py --data-dir ml_data_sf_L16 --model mlp
```

## 4. Main model: 3D CNN

```bash
python3 ml/train_cnn3d.py --data-dir ml_data_L16 --epochs 40 --batch-size 16
python3 ml/evaluate_cnn3d.py --data-dir ml_data_L16
```

Repeat for `L=8` and `L=32` if desired.

## 5. Multi-L CNN plot

```bash
python3 ml/plot_cnn_multi_L.py --dirs ml_data_L8,ml_data_L16,ml_data_L32
```

Main output:

```text
figures/report/cnn3d_P_helical_vs_T_multi_L.pdf
```

## ML transition estimate

For each model, estimate:

```text
Tc_ML where P_helical(T) = 0.5
```

Compare with:

- heat capacity peak
- helical-order fluctuation peak
- drop of helical order
