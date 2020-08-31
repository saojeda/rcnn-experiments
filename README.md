# Application of Deep Learning in Recurrence Plots for Multivariate Nonlinear Time Series Forecasting

## Abstract
We present a framework for multivariate nonlinear time series forecasting that utilizes phase space image representations and deep learning. Recurrence plots (RP) are a phase space visualization tool used for the analysis of dynamical systems. This approach takes advantage of recurrence plots that are used as input image representations for a class of deep learning algorithms called convolutional neural networks. We show that by leveraging recurrence plots with optimal embedding parameters, appropriate representations of underlying dynamics are obtained by the proposed autoregressive deep learning model to produce forecasts.

## Paper (draft)
See manuscript.pdf

## Experiments (last run)
See RCNN_Experiments.ipynb

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
