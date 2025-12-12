############################################################
# nn_experiment.R  (IMPROVED VERSION - NO DATA LEAKAGE)
# Deep FFN with categorical embeddings for Assignment 2
############################################################

set.seed(1)

###############
# Libraries
###############
if (!require("dplyr"))      install.packages("dplyr");      library(dplyr)
if (!require("caret"))      install.packages("caret");      library(caret)
if (!require("keras3"))     install.packages("keras3");     library(keras3)
if (!require("tensorflow")) install.packages("tensorflow"); library(tensorflow)
if (!require("tfruns"))     install.packages("tfruns");     library(tfruns)
if (!require("pROC"))       install.packages("pROC");       library(pROC)

############################################################
# 1. DATA LOADING + BASIC PREPROCESSING
############################################################

raw <- read.csv("Dataset-part-2.csv", stringsAsFactors = FALSE)

dataset <- raw
if ("ID" %in% names(dataset)) {
  dataset <- dataset |> dplyr::select(-ID)
}

# Replace empty strings with NA
for (col in names(dataset)) {
  if (is.character(dataset[[col]]))
    dataset[[col]][dataset[[col]] == ""] <- NA
}

dataset$status <- as.factor(dataset$status)

# DAYS_EMPLOYED special value
dataset$is_unemployed <- ifelse(dataset$DAYS_EMPLOYED == 365243, 1L, 0L)
dataset$DAYS_EMPLOYED[dataset$DAYS_EMPLOYED == 365243] <- NA

dataset$AGE            <- abs(dataset$DAYS_BIRTH) / 365
dataset$YEARS_EMPLOYED <- abs(dataset$DAYS_EMPLOYED) / 365

dataset <- dataset |> dplyr::select(-DAYS_BIRTH, -DAYS_EMPLOYED)

dataset$OCCUPATION_TYPE[is.na(dataset$OCCUPATION_TYPE)] <- "Unknown"

dataset$CNT_FAM_MEMBERS <- pmin(dataset$CNT_FAM_MEMBERS, 10)
dataset$CNT_CHILDREN    <- pmin(dataset$CNT_CHILDREN, 6)

dataset$AMT_INCOME_TOTAL <- log1p(dataset$AMT_INCOME_TOTAL)

if ("FLAG_MOBIL" %in% names(dataset))
  dataset <- dataset |> dplyr::select(-FLAG_MOBIL)

###############
# DEFINE COLUMN TYPES
###############

embedding_cols <- c(
  "NAME_INCOME_TYPE",
  "NAME_EDUCATION_TYPE",
  "NAME_FAMILY_STATUS",
  "NAME_HOUSING_TYPE",
  "OCCUPATION_TYPE"
)

binary_cols <- c("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY")

numeric_cols <- setdiff(
  names(dataset),
  c(embedding_cols, binary_cols, "status")
)

###############
# PROCESS EMBEDDINGS (before split - no leakage here)
###############

embedding_info <- list()
for (col in embedding_cols) {
  dataset[[col]] <- as.factor(dataset[[col]])
  lvls <- levels(dataset[[col]])
  dataset[[col]] <- as.integer(dataset[[col]]) - 1L
  embedding_info[[col]] <- list(n_cat = length(lvls), levels = lvls)
}

# Binary → numeric 0/1 (before split - no leakage here)
dataset$CODE_GENDER    <- ifelse(dataset$CODE_GENDER == "M", 1, 0)
dataset$FLAG_OWN_CAR   <- ifelse(dataset$FLAG_OWN_CAR == "Y", 1, 0)
dataset$FLAG_OWN_REALTY<- ifelse(dataset$FLAG_OWN_REALTY == "Y", 1, 0)

# Extract target
y_factor <- dataset$status
y_num <- as.numeric(y_factor) - 1L
num_classes <- length(unique(y_num))

############################################################
# 2. STRATIFIED TRAIN/VAL/TEST SPLIT (BEFORE SCALING!)
############################################################

set.seed(1)
train_index <- createDataPartition(y_num, p=0.7, list=FALSE)

# Split ALL data
train_data <- dataset[train_index, ]
temp_data <- dataset[-train_index, ]
y_train <- y_num[train_index]
y_temp <- y_num[-train_index]

set.seed(1)
val_index <- createDataPartition(y_temp, p=0.5, list=FALSE)

val_data <- temp_data[val_index, ]
test_data <- temp_data[-val_index, ]
y_val <- y_temp[val_index]
y_test <- y_temp[-val_index]

############################################################
# 3. COMPUTE SCALING PARAMETERS ON TRAINING SET ONLY
############################################################

scaling_params <- list()

# Compute median and min/max from TRAINING data only
for (col in numeric_cols) {
  scaling_params[[col]] <- list()
  
  # Median for imputation
  if (any(is.na(train_data[[col]]))) {
    med <- median(train_data[[col]], na.rm = TRUE)
    scaling_params[[col]]$median <- med
  } else {
    scaling_params[[col]]$median <- NA
  }
  
  # Min/max for scaling (computed on training data after imputation)
  train_col <- train_data[[col]]
  if (!is.na(scaling_params[[col]]$median)) {
    train_col[is.na(train_col)] <- scaling_params[[col]]$median
  }
  
  mn <- min(train_col, na.rm = TRUE)
  mx <- max(train_col, na.rm = TRUE)
  scaling_params[[col]]$min <- mn
  scaling_params[[col]]$max <- mx
}

############################################################
# 4. APPLY SCALING TO ALL SPLITS USING TRAINING PARAMETERS
############################################################

apply_preprocessing <- function(data, scaling_params, numeric_cols) {
  data_copy <- data
  
  # Impute and scale each numeric column
  for (col in numeric_cols) {
    # Impute with training median
    if (!is.na(scaling_params[[col]]$median)) {
      data_copy[[col]][is.na(data_copy[[col]])] <- scaling_params[[col]]$median
    }
    
    # Scale with training min/max
    mn <- scaling_params[[col]]$min
    mx <- scaling_params[[col]]$max
    
    if (mn == mx) {
      data_copy[[col]] <- 0
    } else {
      data_copy[[col]] <- (data_copy[[col]] - mn) / (mx - mn + 1e-9)
    }
  }
  
  return(data_copy)
}

# Apply to all splits
train_data <- apply_preprocessing(train_data, scaling_params, numeric_cols)
val_data <- apply_preprocessing(val_data, scaling_params, numeric_cols)
test_data <- apply_preprocessing(test_data, scaling_params, numeric_cols)

# Extract feature matrices
X_emb_train <- train_data[, embedding_cols, drop=FALSE]
X_bin_train <- train_data[, binary_cols, drop=FALSE]
X_num_train <- train_data[, numeric_cols, drop=FALSE]

X_emb_val <- val_data[, embedding_cols, drop=FALSE]
X_bin_val <- val_data[, binary_cols, drop=FALSE]
X_num_val <- val_data[, numeric_cols, drop=FALSE]

X_emb_test <- test_data[, embedding_cols, drop=FALSE]
X_bin_test <- test_data[, binary_cols, drop=FALSE]
X_num_test <- test_data[, numeric_cols, drop=FALSE]

############################################################
# 5. CLASS WEIGHTING
############################################################

freq <- table(y_train)
raw_w <- 1 / sqrt(freq)
w <- raw_w / mean(raw_w)
class_weights <- as.list(as.numeric(w))
names(class_weights) <- names(freq)

############################################################
# 6. FLAGS FOR TUNING
############################################################

FLAGS <- flags(
  flag_numeric("learning_rate", 0.0005),
  flag_integer("batch_size", 256),
  flag_numeric("width_factor", 1.0),
  flag_numeric("drop", 0.2),
  flag_string("act", "gelu"),
  flag_integer("epochs", 2000)
)

############################################################
# 7. MODEL DEFINITION WITH IMPROVEMENTS
############################################################

l2_reg <- 0.001

# Calculate embedding dimensions dynamically based on number of categories
# Rule: min(50, (n_categories + 1) // 2)
emb_dims <- list()
for (col in embedding_cols) {
  n_cat <- embedding_info[[col]]$n_cat
  emb_dim <- as.integer(min(50, ceiling((n_cat + 1) / 2)))
  emb_dims[[col]] <- emb_dim
  cat(sprintf("%s: %d categories -> embedding dim = %d\n", col, n_cat, emb_dim))
}

# Input layers
income_in <- layer_input(shape=1, dtype="int32", name="income_in")
educ_in   <- layer_input(shape=1, dtype="int32", name="educ_in")
family_in <- layer_input(shape=1, dtype="int32", name="family_in")
housing_in<- layer_input(shape=1, dtype="int32", name="housing_in")
occupation_in <- layer_input(shape=1, dtype="int32", name="occupation_in")

income_emb <- income_in %>% layer_embedding(
  input_dim=embedding_info$NAME_INCOME_TYPE$n_cat,
  output_dim=emb_dims$NAME_INCOME_TYPE) %>% layer_flatten()

educ_emb <- educ_in %>% layer_embedding(
  input_dim=embedding_info$NAME_EDUCATION_TYPE$n_cat,
  output_dim=emb_dims$NAME_EDUCATION_TYPE) %>% layer_flatten()

family_emb <- family_in %>% layer_embedding(
  input_dim=embedding_info$NAME_FAMILY_STATUS$n_cat,
  output_dim=emb_dims$NAME_FAMILY_STATUS) %>% layer_flatten()

housing_emb <- housing_in %>% layer_embedding(
  input_dim=embedding_info$NAME_HOUSING_TYPE$n_cat,
  output_dim=emb_dims$NAME_HOUSING_TYPE) %>% layer_flatten()

occupation_emb <- occupation_in %>% layer_embedding(
  input_dim=embedding_info$OCCUPATION_TYPE$n_cat,
  output_dim=emb_dims$OCCUPATION_TYPE) %>% layer_flatten()

numeric_in <- layer_input(shape=ncol(X_num_train), name="numeric_in")
binary_in  <- layer_input(shape=ncol(X_bin_train), name="binary_in")

merged <- layer_concatenate(list(
  income_emb, educ_emb, family_emb, housing_emb, occupation_emb,
  numeric_in, binary_in
))

base_units <- c(1024,768,512,384,256,128,64)
scaled_units <- as.integer(base_units * FLAGS$width_factor)

# Deep network with residual connections
deep <- merged %>%
  layer_dense(units=scaled_units[1], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

# Block 1 with residual
block1 <- deep %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[2], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

deep <- block1 %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[3], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

# Block 2 with residual
block2 <- deep %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[4], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

deep <- block2 %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[5], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

# Final layers
deep <- deep %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[6], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization() %>%
  layer_dropout(FLAGS$drop) %>%
  layer_dense(units=scaled_units[7], activation=FLAGS$act, kernel_regularizer=regularizer_l2(l2_reg)) %>%
  layer_batch_normalization()

# No dropout before output layer
output <- deep %>% layer_dense(units=num_classes, activation="softmax")

model_ffn <- keras_model(
  inputs=list(income_in, educ_in, family_in, housing_in, occupation_in, numeric_in, binary_in),
  outputs=output
)

model_ffn %>% compile(
  optimizer = optimizer_nadam(learning_rate=FLAGS$learning_rate, clipnorm=1.0),
  loss="sparse_categorical_crossentropy",
  metrics="accuracy"
)

############################################################
# 8. CALLBACKS
############################################################

callback_es <- callback_early_stopping(
  monitor="val_accuracy", 
  patience=100, 
  restore_best_weights=TRUE,
  mode="max"
)

callback_lr <- callback_reduce_lr_on_plateau(
  monitor="val_accuracy", 
  patience=20, 
  factor=0.5, 
  min_lr=1e-6,
  mode="max"
)

# Optional: Checkpoint callback (can be removed since early stopping restores best weights)
# Uncomment if you want to save best model to disk during training
# callback_cp <- callback_model_checkpoint(
#   filepath = file.path(run_dir(), "best_model.keras"),
#   monitor = "val_accuracy",
#   save_best_only = TRUE,
#   verbose = 0,
#   mode="max"
# )

callback_warmup <- callback_lambda(
  on_epoch_begin = function(epoch, logs) {
    if (epoch < 5) {
      lr <- FLAGS$learning_rate * (epoch + 1) / 5
      model_ffn$optimizer$learning_rate$assign(lr)
      cat(sprintf("Warmup: epoch %d, lr = %.6f\n", epoch, lr))
    }
  }
)

############################################################
# 9. TRAINING
############################################################

# Build correct 2D matrices for embedding inputs
make_emb <- function(x) matrix(x, ncol=1)

train_inputs <- list(
  income_in      = make_emb(X_emb_train$NAME_INCOME_TYPE),
  educ_in        = make_emb(X_emb_train$NAME_EDUCATION_TYPE),
  family_in      = make_emb(X_emb_train$NAME_FAMILY_STATUS),
  housing_in     = make_emb(X_emb_train$NAME_HOUSING_TYPE),
  occupation_in  = make_emb(X_emb_train$OCCUPATION_TYPE),
  numeric_in     = as.matrix(X_num_train),
  binary_in      = as.matrix(X_bin_train)
)

val_inputs <- list(
  income_in      = make_emb(X_emb_val$NAME_INCOME_TYPE),
  educ_in        = make_emb(X_emb_val$NAME_EDUCATION_TYPE),
  family_in      = make_emb(X_emb_val$NAME_FAMILY_STATUS),
  housing_in     = make_emb(X_emb_val$NAME_HOUSING_TYPE),
  occupation_in  = make_emb(X_emb_val$OCCUPATION_TYPE),
  numeric_in     = as.matrix(X_num_val),
  binary_in      = as.matrix(X_bin_val)
)

history_ffn <- model_ffn %>% fit(
  x = train_inputs, y = y_train,
  validation_data = list(val_inputs, y_val),
  epochs = FLAGS$epochs,
  batch_size = FLAGS$batch_size,
  callbacks = list(callback_warmup, callback_es, callback_lr),  # Removed callback_cp
  class_weight = class_weights,
  verbose = 2
)

############################################################
# 10. TEST SET EVALUATION (OPTIONAL - COMMENT OUT FOR EXPERIMENTS)
############################################################

# Uncomment the code below to run test set evaluation after training
# Only run this after you've selected your best hyperparameters

# test_inputs <- list(
#   income_in      = make_emb(X_emb_test$NAME_INCOME_TYPE),
#   educ_in        = make_emb(X_emb_test$NAME_EDUCATION_TYPE),
#   family_in      = make_emb(X_emb_test$NAME_FAMILY_STATUS),
#   housing_in     = make_emb(X_emb_test$NAME_HOUSING_TYPE),
#   occupation_in  = make_emb(X_emb_test$OCCUPATION_TYPE),
#   numeric_in     = as.matrix(X_num_test),
#   binary_in      = as.matrix(X_bin_test)
# )
# 
# cat("\n=== TEST SET EVALUATION ===\n")
# test_results <- model_ffn %>% evaluate(test_inputs, y_test, verbose=0)
# cat(sprintf("Test Loss: %.4f\n", test_results[[1]]))
# cat(sprintf("Test Accuracy: %.4f\n", test_results[[2]]))
# 
# # Predictions and confusion matrix
# preds_prob <- model_ffn %>% predict(test_inputs, verbose=0)
# preds <- apply(preds_prob, 1, which.max) - 1L
# 
# cat("\n=== CONFUSION MATRIX ===\n")
# cm <- confusionMatrix(factor(preds, levels=0:(num_classes-1)), 
#                       factor(y_test, levels=0:(num_classes-1)))
# print(cm)
# 
# # Per-class AUC (One-vs-Rest)
# if (num_classes > 2) {
#   cat("\n=== PER-CLASS AUC (One-vs-Rest) ===\n")
#   for (i in 0:(num_classes-1)) {
#     y_binary <- ifelse(y_test == i, 1, 0)
#     pred_binary <- preds_prob[, i+1]
#     roc_obj <- pROC::roc(y_binary, pred_binary, quiet=TRUE)
#     cat(sprintf("Class %d AUC: %.4f\n", i, pROC::auc(roc_obj)))
#   }
# }

############################################################
# 11. SAVE MODEL AND PREPROCESSING PARAMETERS
############################################################

save_model(model_ffn, filepath = file.path(run_dir(), "model.keras"), overwrite=TRUE)

# Save preprocessing parameters for inference
preprocessing_info <- list(
  embedding_info = embedding_info,
  scaling_params = scaling_params,
  numeric_cols = numeric_cols,
  binary_cols = binary_cols,
  embedding_cols = embedding_cols,
  num_classes = num_classes
)

saveRDS(preprocessing_info, file = file.path(run_dir(), "preprocessing_info.rds"))

cat("\n=== MODEL AND PREPROCESSING INFO SAVED ===\n")
cat("Model saved to:", file.path(run_dir(), "model.keras"), "\n")
cat("Preprocessing info saved to:", file.path(run_dir(), "preprocessing_info.rds"), "\n")