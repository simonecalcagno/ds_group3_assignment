################################################################################
# nn_experiment.R
# Feed-forward neural network for Assignment 2
# - full preprocessing (capping, Unknown occupation, etc.)
# - class weighting for imbalance
# - FLAGS for tfruns::tuning_run()
################################################################################

set.seed(1)

################################################################################
# Libraries
################################################################################

if (!require("dplyr")) install.packages("dplyr")
library(dplyr)

if (!require("caret")) install.packages("caret")
library(caret)

if (!require("keras3")) install.packages("keras3")
library(keras3)

if (!require("tensorflow")) install.packages("tensorflow")
library(tensorflow)

if (!require("tfruns")) install.packages("tfruns")
library(tfruns)

if (!require("fastDummies")) install.packages("fastDummies")
library(fastDummies)

################################################################################
# 1. Data loading and preprocessing (no leakage)
################################################################################

# Step 1: Load raw data and remove technical ID column
raw <- read.csv("Dataset-part-2.csv", stringsAsFactors = FALSE)
dataset <- raw

if ("ID" %in% names(dataset)) {
  dataset <- dataset %>% select(-ID)
}

# Step 2: Replace empty strings in character columns with NA
for (col in names(dataset)) {
  if (is.character(dataset[[col]])) {
    dataset[[col]][dataset[[col]] == ""] <- NA
  }
}

# Step 3: Target as factor (fix levels once)
dataset$status <- factor(dataset$status)
status_levels <- levels(dataset$status)

# Step 4: Handle special placeholder in DAYS_EMPLOYED
dataset$is_unemployed <- ifelse(dataset$DAYS_EMPLOYED == 365243, 1L, 0L)
dataset$DAYS_EMPLOYED[dataset$DAYS_EMPLOYED == 365243] <- NA

# Step 5: Create age and years employed (in years)
dataset$AGE_YEARS <- abs(dataset$DAYS_BIRTH) / 365
dataset$YEARS_EMPLOYED <- abs(dataset$DAYS_EMPLOYED) / 365

# Optional caps (outlier capping)
dataset$AGE_YEARS <- pmin(dataset$AGE_YEARS, 90)
dataset$YEARS_EMPLOYED <- pmin(dataset$YEARS_EMPLOYED, 60)

# Remove original day-based columns
dataset <- dataset %>% select(-DAYS_BIRTH, -DAYS_EMPLOYED)

# Step 6: Treat missing OCCUPATION_TYPE as its own category
dataset$OCCUPATION_TYPE[is.na(dataset$OCCUPATION_TYPE)] <- "Unknown"

# Step 7: Cap extreme values
dataset$CNT_FAM_MEMBERS <- pmin(dataset$CNT_FAM_MEMBERS, 10)
dataset$CNT_CHILDREN <- pmin(dataset$CNT_CHILDREN, 6)

# Step 9: Remove uninformative constant feature
if ("FLAG_MOBIL" %in% names(dataset)) {
  dataset <- dataset %>% select(-FLAG_MOBIL)
}

# Step 10: Dummy encoding
dataset$status <- factor(dataset$status, levels = status_levels)

cat_cols <- names(dataset)[sapply(dataset, is.character) | sapply(dataset, is.factor)]
cat_cols <- setdiff(cat_cols, "status")

dataset_dummies <- fastDummies::dummy_cols(
  dataset,
  select_columns = cat_cols,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

features <- dataset_dummies %>% select(-status)
y_factor <- dataset_dummies$status
y_num <- as.numeric(y_factor) - 1L
num_classes <- length(unique(y_num))

################################################################################
# 2. Train / validation / test split (stratified)
################################################################################

set.seed(1)
train_index <- createDataPartition(y_num, p = 0.7, list = FALSE)

x_all <- data.matrix(features)

x_train_raw <- x_all[train_index, , drop = FALSE]
y_train <- y_num[train_index]

x_temp_raw <- x_all[-train_index, , drop = FALSE]
y_temp <- y_num[-train_index]

set.seed(1)
val_index <- createDataPartition(y_temp, p = 0.5, list = FALSE)

x_val_raw <- x_temp_raw[val_index, , drop = FALSE]
y_val <- y_temp[val_index]

x_test_raw <- x_temp_raw[-val_index, , drop = FALSE]
y_test <- y_temp[-val_index]

################################################################################
# 3. Imputation + scaling using TRAIN statistics only
################################################################################

# 3.1 Median imputation
medians <- apply(x_train_raw, 2, function(v) {
  m <- median(v, na.rm = TRUE)
  if (is.na(m)) 0 else m
})

impute_median <- function(mat, med) {
  for (j in seq_len(ncol(mat))) {
    idx_na <- is.na(mat[, j])
    if (any(idx_na)) {
      mat[idx_na, j] <- med[j]
    }
  }
  mat
}

x_train_imp <- impute_median(x_train_raw, medians)
x_val_imp   <- impute_median(x_val_raw, medians)
x_test_imp  <- impute_median(x_test_raw, medians)

# 3.2 Min–max scaling
mins <- apply(x_train_imp, 2, min)
maxs <- apply(x_train_imp, 2, max)

same <- which(maxs == mins)
maxs[same] <- mins[same] + 1

scale_minmax <- function(mat, mins, maxs) {
  scaled <- sweep(mat, 2, mins, "-")
  sweep(scaled, 2, (maxs - mins), "/")
}

x_train_scaled <- scale_minmax(x_train_imp, mins, maxs)
x_val  <- scale_minmax(x_val_imp, mins, maxs)
x_test <- scale_minmax(x_test_imp, mins, maxs)

################################################################################
# 3.3 Oversampling (train only)
################################################################################

train_df <- as.data.frame(x_train_scaled)
train_df$status <- factor(y_train)

tab <- table(train_df$status)
prop <- tab / sum(tab)

target_ratio <- 0.15
minor_threshold <- 0.08
max_dup_factor <- 5

majority_class <- names(tab)[which.max(tab)]
majority_n <- max(tab)

resampled_idx <- integer(0)

for (cl in names(tab)) {
  idx <- which(train_df$status == cl)
  n_cl <- length(idx)
  p_cl <- prop[cl]
  
  if (cl == majority_class || p_cl >= minor_threshold) {
    resampled_idx <- c(resampled_idx, idx)
  } else {
    target_n <- min(as.integer(target_ratio * majority_n), n_cl * max_dup_factor)
    extra_idx <- sample(idx, size = max(0, target_n - n_cl), replace = TRUE)
    resampled_idx <- c(resampled_idx, idx, extra_idx)
  }
}

resampled_idx <- sample(resampled_idx)

x_train <- x_train_scaled[resampled_idx, , drop = FALSE]
y_train <- as.numeric(train_df$status[resampled_idx]) - 1L

################################################################################
# 4. Remove constant columns
################################################################################

const_cols <- which(maxs == mins)
if (length(const_cols) > 0) {
  x_train <- x_train[, -const_cols, drop = FALSE]
  x_val   <- x_val[, -const_cols, drop = FALSE]
  x_test  <- x_test[, -const_cols, drop = FALSE]
}

################################################################################
# 5. Hyperparameters via FLAGS
################################################################################

FLAGS <- flags(
  flag_numeric("learning_rate", 0.0005),
  flag_integer("batch_size", 256),
  flag_string("act", "relu"),
  flag_integer("epochs", 1000),
  flag_numeric("l2_reg", 0.001)
)

################################################################################
# 6. Model definition
################################################################################

model_ffn <- keras_model_sequential() %>%
  layer_dense(
    units = 512,
    activation = FLAGS$act,
    input_shape = ncol(x_train)
  ) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = FLAGS$act) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = FLAGS$act) %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = num_classes, activation = "softmax")

y_train_cat <- to_categorical(y_train, num_classes)
y_val_cat   <- to_categorical(y_val, num_classes)

model_ffn %>% compile(
  optimizer = optimizer_sgd(learning_rate = FLAGS$learning_rate),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

summary(model_ffn)

################################################################################
# 7. Training
################################################################################

callback_es <- callback_early_stopping(
  monitor = "val_loss",
  patience = 300,
  restore_best_weights = TRUE,
  min_delta = 0.001
)

callback_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,
  patience = 50,
  min_lr = 1e-4,
  verbose = 1
)

history_ffn <- model_ffn %>% fit(
  x_train,
  y_train_cat,
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  validation_data = list(x_val, y_val_cat),
  callbacks = list(callback_es, callback_lr),
  verbose = 2
)

################################################################################
# 9. Save model
################################################################################

save_model(
  model_ffn,
  filepath = file.path(run_dir(), "model.keras"),
  overwrite = TRUE
)
