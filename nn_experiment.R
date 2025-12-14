############################################################
# nn_experiment.R
# Feed-forward neural network for Assignment 2
# - full preprocessing (capping, Unknown occupation, etc.)
# - class weighting for imbalance
# - FLAGS for tfruns::tuning_run()
############################################################

set.seed(1)

###############
# Libraries
###############
if (!require("dplyr"))       install.packages("dplyr");       library(dplyr)
if (!require("caret"))       install.packages("caret");       library(caret)
if (!require("keras3"))      install.packages("keras3");      library(keras3)
if (!require("tensorflow"))  install.packages("tensorflow");  library(tensorflow)
if (!require("tfruns"))      install.packages("tfruns");      library(tfruns)
if (!require("fastDummies")) install.packages("fastDummies"); library(fastDummies)


############################################################
# 1. Data loading and preprocessing (no leakage)
############################################################

# Step 1: Load raw data and remove technical ID column
raw <- read.csv("Dataset-part-2.csv", stringsAsFactors = FALSE)

dataset <- raw
if ("ID" %in% names(dataset)) {
  dataset <- dataset %>% dplyr::select(-ID)
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

# Step 4: Handle special placeholder in DAYS_EMPLOYED (365243 = unemployed)
dataset$is_unemployed <- ifelse(dataset$DAYS_EMPLOYED == 365243, 1L, 0L)
dataset$DAYS_EMPLOYED[dataset$DAYS_EMPLOYED == 365243] <- NA

# Step 5: Create age and years employed (in years)
dataset$AGE_YEARS      <- abs(dataset$DAYS_BIRTH) / 365
dataset$YEARS_EMPLOYED <- abs(dataset$DAYS_EMPLOYED) / 365

# Optional caps (outlier capping)
dataset$AGE_YEARS      <- pmin(dataset$AGE_YEARS, 90)
dataset$YEARS_EMPLOYED <- pmin(dataset$YEARS_EMPLOYED, 60)

# Remove original day-based columns
dataset <- dataset %>% dplyr::select(-DAYS_BIRTH, -DAYS_EMPLOYED)

# Step 6: Treat missing OCCUPATION_TYPE as its own category "Unknown"
dataset$OCCUPATION_TYPE[is.na(dataset$OCCUPATION_TYPE)] <- "Unknown"

# Step 7: Cap extreme values for children and family size (outlier handling)
dataset$CNT_FAM_MEMBERS <- pmin(dataset$CNT_FAM_MEMBERS, 10)
dataset$CNT_CHILDREN    <- pmin(dataset$CNT_CHILDREN,    6)

# Step 8: Log-transform income to reduce skewness (with outlier capping)
if ("AMT_INCOME_TOTAL" %in% names(dataset)) {
  q99 <- quantile(dataset$AMT_INCOME_TOTAL, 0.99, na.rm = TRUE)
  dataset$AMT_INCOME_TOTAL <- pmin(dataset$AMT_INCOME_TOTAL, q99)
  dataset$AMT_INCOME_TOTAL <- log1p(dataset$AMT_INCOME_TOTAL)
}

# Step 9: Remove uninformative constant feature FLAG_MOBIL (always 1)
if ("FLAG_MOBIL" %in% names(dataset)) {
  dataset <- dataset %>% dplyr::select(-FLAG_MOBIL)
}

# Step 10: Dummy-encode categorical predictors (one-hot)
dataset$status <- factor(dataset$status, levels = status_levels)

cat_cols <- names(dataset)[sapply(dataset, is.character) | sapply(dataset, is.factor)]
cat_cols <- setdiff(cat_cols, "status")  # exclude target

dataset_dummies <- fastDummies::dummy_cols(
  dataset,
  select_columns          = cat_cols,
  remove_first_dummy      = TRUE,   # avoid perfect collinearity
  remove_selected_columns = TRUE    # drop original categorical columns
)

# Separate predictors and target (still NO scaling / imputation yet)
features <- dataset_dummies %>% dplyr::select(-status)
y_factor <- dataset_dummies$status
y_num    <- as.numeric(y_factor) - 1L
num_classes <- length(unique(y_num))

############################################################
# 2. Train / validation / test split (stratified)
############################################################

set.seed(1)
train_index <- createDataPartition(y_num, p = 0.7, list = FALSE)

x_all <- data.matrix(features)

x_train_raw <- x_all[train_index, , drop = FALSE]
y_train     <- y_num[train_index]

x_temp_raw  <- x_all[-train_index, , drop = FALSE]
y_temp      <- y_num[-train_index]

set.seed(1)
val_index <- createDataPartition(y_temp, p = 0.5, list = FALSE)

x_val_raw  <- x_temp_raw[val_index, , drop = FALSE]
y_val      <- y_temp[val_index]

x_test_raw <- x_temp_raw[-val_index, , drop = FALSE]
y_test     <- y_temp[-val_index]



############################################################
# 3. Imputation + scaling using TRAIN statistics only
############################################################

# 3.1 Median imputation (train stats → apply to val/test)
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
x_val_imp   <- impute_median(x_val_raw,   medians)
x_test_imp  <- impute_median(x_test_raw,  medians)

############################################################
# 3.2 Scale to [0,1] using train min/max (no leakage)
############################################################

mins  <- apply(x_train_imp, 2, min)
maxs  <- apply(x_train_imp, 2, max)
range <- maxs - mins
range[range == 0] <- 1  # avoid div-by-zero for constant columns

scale_minmax <- function(mat, mins, range) {
  scaled <- sweep(mat, 2, mins, FUN = "-")
  scaled <- sweep(scaled, 2, range, FUN = "/")
  scaled[is.na(scaled)]       <- 0
  scaled[is.infinite(scaled)] <- 0
  scaled
}

# Scale train/val/test
x_train_scaled <- scale_minmax(x_train_imp, mins, range)
x_val          <- scale_minmax(x_val_imp,   mins, range)
x_test         <- scale_minmax(x_test_imp,  mins, range)

############################################################
# 3.3 Thoughtful oversampling on *scaled* training data
#     (train only, rare classes only, limited factor)
############################################################

# We'll oversample on the scaled training data.
train_df <- as.data.frame(x_train_scaled)
train_df$status <- factor(y_train)  # still 0..K-1 but as factor

cat("Class distribution BEFORE oversampling:\n")
print(table(train_df$status))

# --- config knobs (you can tweak these) ---
minor_threshold <- 0.05   # oversample classes with < 5% of train
target_ratio    <- 0.15   # oversample to 15% of majority size
max_dup_factor  <- 5      # never duplicate a class more than 5x its original size
# -----------------------------------------

tab   <- table(train_df$status)
n_tot <- sum(tab)
prop  <- tab / n_tot

majority_class <- names(tab)[which.max(tab)]
majority_n     <- max(tab)

cat("\nProportions per class:\n")
print(round(prop, 4))
cat("\nMajority class:", majority_class, "with", majority_n, "samples\n")

resampled_idx <- integer(0)

for (cl in names(tab)) {
  idx   <- which(train_df$status == cl)
  n_cl  <- length(idx)
  p_cl  <- prop[cl]
  
  # Step 2: identify which classes need help
  if (cl == majority_class || p_cl >= minor_threshold) {
    # Keep majority and medium-frequency classes as they are
    resampled_idx <- c(resampled_idx, idx)
  } else {
    # Rare class → candidate for oversampling
    # Step 3: how much to oversample
    target_n <- as.integer(target_ratio * majority_n)
    
    # respect max duplication factor
    max_allowed <- n_cl * max_dup_factor
    target_n <- min(target_n, max_allowed)
    
    if (target_n <= n_cl) {
      # Already at/above target → do nothing
      resampled_idx <- c(resampled_idx, idx)
    } else {
      # Controlled random oversampling with replacement
      extra_idx <- sample(idx, size = target_n - n_cl, replace = TRUE)
      resampled_idx <- c(resampled_idx, idx, extra_idx)
    }
  }
}

# Shuffle to avoid any ordering bias
resampled_idx <- sample(resampled_idx)

# Final training data after oversampling
x_train <- x_train_scaled[resampled_idx, , drop = FALSE]
y_train <- as.numeric(train_df$status[resampled_idx]) - 1L  # back to 0..K-1

cat("\nClass distribution AFTER oversampling:\n")
print(table(factor(y_train, levels = 0:(num_classes - 1))))

############################################################
# 3.4 Remove constant columns (based on original mins/maxs)
############################################################

const_cols <- which(maxs == mins)
if (length(const_cols) > 0) {
  cat("Removing", length(const_cols), "constant columns\n")
  x_train <- x_train[, -const_cols, drop = FALSE]
  x_val   <- x_val[,   -const_cols, drop = FALSE]
  x_test  <- x_test[,  -const_cols, drop = FALSE]
}


############################################################
# 4. Class weights to handle imbalance
############################################################

# freq  <- table(y_train)
# raw_w <- 1 / sqrt(freq)      # softer than 1/freq
# w     <- raw_w / mean(raw_w) # normalise around 1
# 
# class_weights <- as.list(as.numeric(w))
# names(class_weights) <- names(freq)
# print(class_weights)

############################################################
# 5. Hyperparameters via FLAGS (for tfruns::tuning_run)
############################################################

FLAGS <- flags(
  flag_numeric("learning_rate", 0.0005),
  flag_integer("batch_size",    256),
  flag_numeric("width_factor",  1.0),    # scales layer widths
  flag_string ("act",           "relu"), # activation for all hidden layers
  flag_numeric("drop",          0.2),    # dropout rate for all hidden layers
  flag_integer("epochs",        1000),
  flag_numeric("l2_reg",        0.001)
)

############################################################
# 6. Model definition (feed-forward network, 4 hidden layers)
############################################################

l2_reg <- FLAGS$l2_reg

base_units  <- c(1024, 512, 256, 128, 64)
units_scaled <- as.integer(base_units * FLAGS$width_factor)

model_ffn <- keras_model_sequential() %>%
  layer_dense(
    units             = units_scaled[1],
    activation        = FLAGS$act,
    input_shape       = ncol(x_train),
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate  = FLAGS$drop) %>%
  
  layer_dense(
    units             = units_scaled[2],
    activation        = FLAGS$act,
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate  = FLAGS$drop) %>%
  
  layer_dense(
    units             = units_scaled[3],
    activation        = FLAGS$act,
    kernel_regularizer =regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate  = FLAGS$drop) %>%
  
  layer_dense(
    units             = units_scaled[4],
    activation        = FLAGS$act,
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate  = FLAGS$drop) %>%
  
  
  layer_dense(
    units      = num_classes,
    activation = "softmax"
  )

model_ffn %>% compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learning_rate),
  loss      = "sparse_categorical_crossentropy",
  metrics   = "accuracy"
)

summary(model_ffn)

############################################################
# 7. Training with early stopping + LR scheduling
############################################################

callback_es <- callback_early_stopping(
  monitor              = "val_loss",
  patience             = 300,
  restore_best_weights = TRUE
)

callback_lr <- callback_reduce_lr_on_plateau(
  monitor  = "val_loss",
  factor   = 0.75,   # or at most 0.3
  patience = 50,    # smaller than your early stopping patience
  min_lr   = 1e-6,
  verbose  = 1
)

history_ffn <- model_ffn %>% fit(
  x_train,
  y_train,
  epochs          = FLAGS$epochs,
  batch_size      = FLAGS$batch_size,
  validation_data = list(x_val, y_val),
  callbacks       = list(callback_es, callback_lr),
  #class_weight    = class_weights,
  verbose         = 2
)

###########################################################
# 8. OPTIONAL: Evaluation on test set + confusion matrix
#    IMPORTANT: keep this commented during tuning_run().
#    Uncomment only when you train the FINAL best model.
############################################################

# library(caret)  # already loaded at top

# --- scalar metrics on test set ---
# scores <- model_ffn %>% evaluate(
#   x_test,
#   y_test,
#   verbose = 0
# )
# cat("\nTest loss:", scores["loss"],
#     "  Test accuracy:", scores["accuracy"], "\n")

# --- confusion matrix on test set ---
# 1. Predict class probabilities
# y_test_pred_prob <- model_ffn %>% predict(x_test)
#
# 2. Convert to class index 0..(num_classes-1)
# (keras gives 1-based argmax, so subtract 1)
# y_test_pred_class <- apply(y_test_pred_prob, 1, which.max) - 1L
#
# 3. Map numeric classes back to factor labels
#    status_levels was defined at the top from dataset$status
# true_labels <- factor(
#   status_levels[y_test + 1L],
#   levels = status_levels
# )
# pred_labels <- factor(
#   status_levels[y_test_pred_class + 1L],
#   levels = status_levels
# )
#
# 4. Confusion matrix
# cm <- confusionMatrix(pred_labels, true_labels)
# print(cm)

############################################################
# 9. Save model in this run's directory
############################################################

save_model(
  model_ffn,
  filepath  = file.path(run_dir(), "model.keras"),
  overwrite = TRUE
)
