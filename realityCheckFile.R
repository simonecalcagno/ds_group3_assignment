## ================== Reality Check Script ==================
## Assignment 1 – Reality Check

## 1) Specify the file name of the (secret) dataset

secret_file <- "LCdata_sample1000.csv"


## 2) Load required packages and the preprocessing function
library(readr)
library(dplyr)

source("preprocess_function.R")


## 3) Load the secret dataset
raw_secret <- read_delim(
  secret_file,
  delim = ";",
  na = c("", "NA", "N/A")
)

cat("Loaded secret data with", nrow(raw_secret), "rows and",
    ncol(raw_secret), "columns\n")


## 4) Apply the SAME preprocessing as used during training
secret_prep <- preprocess_lc(raw_secret)

cat("After preprocessing:", nrow(secret_prep), "rows and",
    ncol(secret_prep), "columns\n")


## 5) Load the saved best model (XGBoost)

model_objects <- readRDS("best_model_assignment1.rds")

xgb_model <- model_objects$model
feat_names <- model_objects$features


cat("Loaded XGBoost model trained on", length(feat_names), "features\n")
## 7) Align columns to training features using saved 'feat_names'

# Add missing columns (present in training, missing here) as zeros
missing_cols <- setdiff(feat_names, colnames(X_secret))
if (length(missing_cols) > 0) {
  cat("Adding", length(missing_cols), "missing columns (set to 0)\n")
  for (m in missing_cols) {
    X_secret <- cbind(X_secret, tmp = 0)
    colnames(X_secret)[ncol(X_secret)] <- m
  }
}

# Remove any extra columns not seen during training
extra_cols <- setdiff(colnames(X_secret), feat_names)
if (length(extra_cols) > 0) {
  cat("Dropping", length(extra_cols), "extra columns not used in training\n")
  X_secret <- X_secret[, !(colnames(X_secret) %in% extra_cols), drop = FALSE]
}

# Reorder columns to match training order exactly
X_secret <- X_secret[, feat_names, drop = FALSE]

cat("Aligned model matrix has", ncol(X_secret), "columns (should equal length(feat_names))\n")


## 7) Predict interest rates on the secret data
pred_secret <- predict(xgb_model, newdata = X_secret)

cat("Computed predictions for", length(pred_secret), "observations\n")


## 8) Compute MSE / RMSE if int_rate is available in the secret data
if ("int_rate" %in% names(secret_prep)) {
  mse_secret  <- mean((secret_prep$int_rate - pred_secret)^2, na.rm = TRUE)
  rmse_secret <- sqrt(mse_secret)
  
  cat("Reality-check MSE :", round(mse_secret, 3), "\n")
  cat("Reality-check RMSE:", round(rmse_secret, 3), "\n")
} else {
  cat("Column 'int_rate' not present in secret data -> skipping MSE/RMSE calculation\n")
}

## ================== End of Reality Check Script ==================