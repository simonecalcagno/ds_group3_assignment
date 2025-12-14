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
if (!require("smotefamily")) install.packages("smotefamily"); library(smotefamily)


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

# Step 8: Log-transform income
dataset$AMT_INCOME_TOTAL <- log1p(dataset$AMT_INCOME_TOTAL)

# NEW FEATURES (AFTER log transform)
dataset$INCOME_PER_PERSON <- dataset$AMT_INCOME_TOTAL / 
  pmax(dataset$CNT_FAM_MEMBERS, 1)

dataset$CHILD_RATIO <- dataset$CNT_CHILDREN / 
  pmax(dataset$CNT_FAM_MEMBERS, 1)


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



#############################################################
# 3.1 Median imputation (train stats)
############################################################
medians <- apply(x_train_raw, 2, function(v) {
  m <- median(v, na.rm = TRUE)
  if (is.na(m)) 0 else m
})

impute_median <- function(mat, med) {
  for (j in seq_len(ncol(mat))) {
    idx_na <- is.na(mat[, j])
    if (any(idx_na)) mat[idx_na, j] <- med[j]
  }
  mat
}

x_train_imp <- impute_median(x_train_raw, medians)
x_val_imp   <- impute_median(x_val_raw,   medians)
x_test_imp  <- impute_median(x_test_raw,  medians)

############################################################
# 3.2 Scale BEFORE SMOTE (using train stats)
############################################################

mins  <- apply(x_train_imp, 2, min)
maxs  <- apply(x_train_imp, 2, max)
range <- maxs - mins
range[range == 0] <- 1

scale_minmax <- function(mat, mins, range) {
  scaled <- sweep(mat, 2, mins, FUN = "-")
  scaled <- sweep(scaled, 2, range, FUN = "/")
  scaled[is.na(scaled)]       <- 0
  scaled[is.infinite(scaled)] <- 0
  scaled
}

x_train_scaled <- scale_minmax(x_train_imp, mins, range)
x_val_scaled   <- scale_minmax(x_val_imp,   mins, range)
x_test_scaled  <- scale_minmax(x_test_imp,  mins, range)

############################################################
# 3.25 Remove constant columns (BEFORE SMOTE)
############################################################

const_cols <- which(range == 0)

if (length(const_cols) > 0) {
  cat("Removing", length(const_cols), "constant columns BEFORE SMOTE\n")
  x_train_scaled <- x_train_scaled[, -const_cols, drop = FALSE]
  x_val_scaled   <- x_val_scaled[,   -const_cols, drop = FALSE]
  x_test_scaled  <- x_test_scaled[,  -const_cols, drop = FALSE]
}

############################################################
# 3.3 SMOTE on *scaled* training data
############################################################

############################################################
# 3.3 SMOTE on *scaled* training data (SAFE)
############################################################

train_df <- as.data.frame(x_train_scaled)

# FIX class levels ONCE
train_df$status <- factor(
  y_train,
  levels = 0:(num_classes - 1)
)

cat("Class distribution BEFORE SMOTE:\n")
print(table(train_df$status))

# Apply SMOTE
smote_result <- SMOTE(
  X        = train_df[, -ncol(train_df)],
  target   = train_df$status,
  K        = 5,
  dup_size = 2
)

# Extract data
smote_data <- smote_result$data

# 🔒 FORCE correct levels AFTER SMOTE
smote_data$class <- factor(
  smote_data$class,
  levels = 0:(num_classes - 1)
)

# Extract features
x_train <- as.matrix(smote_data[, -ncol(smote_data)])

# FIX: SMOTE labels start at 1 → convert to 0..(K-1)
y_train <- as.integer(smote_data$class) - 1L

# FINAL feature matrices (must match x_train exactly)
x_val  <- x_val_scaled
x_test <- x_test_scaled

# SAFETY CHECKS
stopifnot(!any(is.na(y_train)))
stopifnot(min(y_train) == 0)
stopifnot(max(y_train) == (num_classes - 1))

cat("Class distribution AFTER SMOTE:\n")
print(table(y_train))



############################################################
# 4. Class weights to handle imbalance
############################################################

 freq  <- table(y_train)
 raw_w <- 1 / sqrt(freq)      # softer than 1/freq
 w     <- raw_w / mean(raw_w) # normalise around 1
 
 class_weights <- as.list(as.numeric(w))
 names(class_weights) <- names(freq)
 print(class_weights)

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

base_units  <- c(512, 256, 128, 64, 32)
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
    units             = units_scaled[5],
    activation        = FLAGS$act) %>%
  
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
  patience             = 200,
  restore_best_weights = TRUE
)

callback_lr <- callback_reduce_lr_on_plateau(
  monitor  = "val_loss",
  factor   = 0.8,   # or at most 0.3
  patience = 100,    # smaller than your early stopping patience
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
  class_weight    = class_weights,
  verbose         = 2
)

############################################################
# 8. (OPTIONAL) Evaluation on test set
#    Commented out so test is NOT used in every tuning run
############################################################

# scores <- model_ffn %>% evaluate(
#   x_test,
#   y_test,
#   verbose = 0
# )
#
# print(scores)

############################################################
# 9. Save model in this run's directory
############################################################

save_model(
  model_ffn,
  filepath  = file.path(run_dir(), "model.keras"),
  overwrite = TRUE
)
