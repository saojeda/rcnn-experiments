# rcnn-experiments

## Paper
[Overleaf](https://www.overleaf.com/6989937774bmvncgjkysgb)

## Experiments (last run)
[Colab notebook](https://colab.research.google.com/drive/1IljUzQuDRDmjE7bmlihuc6L1-tgog6WP?usp=sharing)

## Results
- rcnn/results.csv (raw)
- rcnn/results_agg.csv (aggregated results)

## Statistical tests
- rcnn/bdstest.out (BDS test)
- nemenyi/nemenyi.out (Friedman & Nemenyi tests)

## Dependencies
- numba==0.50.1
- scipy==1.4.1
- pandas==1.0.3
- tensorflow==2.2.0
- numpy==1.18.1
- scikit-learn==0.22.1
- pyts==0.11.0
- statsmodels==0.11.1
- pmdarima==1.7.0

Adapted code from https://github.com/gabrieljaguiar/nemenyi and https://github.com/ChangWeiTan/TSRegression
