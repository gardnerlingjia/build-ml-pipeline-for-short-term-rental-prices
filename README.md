# Build an ML Pipeline for Short-Term Rental Prices in NYC
You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Submission Links

- GitHub Repository: https://github.com/gardnerlingjia/build-ml-pipeline-for-short-term-rental-prices
- Public W&B Project: https://wandb.ai/gardner-lingjia-cariad/nyc_airbnb

## Train the Model on a New Data Sample

1.0.0 = before the NYC boundary cleaning fix, fails on sample2.csv
1.0.1 (or 1.0.2) = first release containing the fix in basic_cleaning
1.0.11 = latest release in the final submitted repo
Release `1.0.0` represents the pipeline before the NYC-boundary cleaning fix and fails on `sample2.csv`. The cleaning fix was introduced in release `1.0.1` (also present in later releases), where `basic_cleaning` filters listings outside the NYC latitude/longitude bounds. The final submitted repository is at release `1.0.11`, which includes this fix and runs successfully on `sample2.csv`, producing a new trained model.

## Project Summary

This project builds an end-to-end machine learning pipeline that:

- downloads and versions the dataset
- cleans the data and removes outliers/duplicates
- validates the data
- splits the dataset into train/validation and test sets
- trains a random forest model
- performs hyperparameter search with Hydra
- selects the best model and tags it as `prod`
- evaluates the selected model on the test set

## Notes

The final submitted model artifact is stored in W&B as an MLflow-exported sklearn model artifact.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/gardnerlingjia/build-ml-pipeline-for-short-term-rental-prices.git
   cd build-ml-pipeline-for-short-term-rental-prices

2. Create and activate the conda environment:
   conda env create -f conda.yml
   conda activate nyc_airbnb_dev

3. Run the full pipeline:
   python main.py

4. Run the random forest hyperparameter sweep:
   python main.py --multirun main.steps=train_random_forest modeling.random_forest.max_depth=5,10,15         modeling.random_forest.n_estimators=50,100

5. Check results in the public W&B project:
   https://wandb.ai/gardner-lingjia-cariad/nyc_airbnb






