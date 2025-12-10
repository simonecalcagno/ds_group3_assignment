############################################################
# nn_experiment.R
# Feed-forward neural network for Assignment 2
# - full preprocessing
# - class weighting for imbalance
# - FLAGS for tfruns::tuning_run()
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
if (!require("fastDummies")) install.packages("fastDummies"); library(fastDummies)


############################################################
# 1. Data loading and preprocessing
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

# Step 3: Convert target column 'status' to factor
dataset$status <- as.factor(dataset$status)

# Step 4: Handle special placeholder in DAYS_EMPLOYED (365243 = unemployed)
dataset$is_unemployed <- ifelse(dataset$DAYS_EMPLOYED == 365243, 1, 0)
dataset$DAYS_EMPLOYED[dataset$DAYS_EMPLOYED == 365243] <- NA

# Step 5: Create age and years employed (in years)
dataset$AGE            <- abs(dataset$DAYS_BIRTH) / 365
dataset$YEARS_EMPLOYED <- abs(dataset$DAYS_EMPLOYED) / 365

# Remove original day-based columns
dataset <- dataset %>% dplyr::select(-DAYS_BIRTH, -DAYS_EMPLOYED)

# Step 6: Treat missing OCCUPATION_TYPE as its own category
dataset$OCCUPATION_TYPE[is.na(dataset$OCCUPATION_TYPE)] <- "Unknown"

# Step 7: Cap extreme values in family members and children
dataset$CNT_FAM_MEMBERS <- pmin(dataset$CNT_FAM_MEMBERS, 10)
dataset$CNT_CHILDREN    <- pmin(dataset$CNT_CHILDREN, 6)

# Step 8: Log-transform income to reduce skewness
dataset$AMT_INCOME_TOTAL <- log1p(dataset$AMT_INCOME_TOTAL)

# Step 9: Remove uninformative constant feature FLAG_MOBIL (always 1)
if ("FLAG_MOBIL" %in% names(dataset)) {
  dataset <- dataset %>% dplyr::select(-FLAG_MOBIL)
}

# Step 10: Separate predictors and target
# features <- dataset %>% dplyr::select(-status)

# Step 11: Encode character predictors as numeric codes
#for (col in names(features)) {
 #  if (is.character(features[[col]])) {
 #    features[[col]] <- as.numeric(as.factor(features[[col]]))
#   }
# }

# Step 12: Median imputation for remaining numeric NAs
#for (col in names(features)) {
#  if (is.numeric(features[[col]])) {
#    if (any(is.na(features[[col]]))) {
#      med <- median(features[[col]], na.rm = TRUE)
#      if (is.na(med)) {
#        features[[col]] <- 0
#      } else {
#        features[[col]][is.na(features[[col]])] <- med
#      }
#    }
#  }
#}

# Step 13: Scale all numeric predictors to [0,1]
#scale_to_zero_one <- function(x) {
#  if (!is.numeric(x)) return(x)
#  if (all(is.na(x)))  return(rep(0, length(x)))
#  mn <- min(x, na.rm = TRUE)
#  mx <- max(x, na.rm = TRUE)
#  if (mn == mx) return(rep(0, length(x)))
#  scaled <- (x - mn) / (mx - mn)
#  scaled[is.na(scaled)]      <- 0
#  scaled[is.infinite(scaled)] <- 0
#  scaled
#}

#features_scaled <- as.data.frame(lapply(features, scale_to_zero_one))
#features_scaled_num <- as.data.frame(lapply(features_scaled, as.numeric))

# Step 14: Final x matrix and numeric y vector
# x <- data.matrix(features_scaled_num)
# y_factor <- dataset$status
# y_num <- as.numeric(y_factor) - 1     # 0-based labels for sparse_categorical

# num_classes <- length(unique(y_num))

# Step 10: Separate predictors and target
# (we'll dummy-encode on the full dataset including status)
dataset$status <- as.factor(dataset$status)

# identify categorical columns (character or factor), except target
cat_cols <- names(dataset)[sapply(dataset, is.character) | sapply(dataset, is.factor)]
cat_cols <- setdiff(cat_cols, "status")

# Step 11: Dummy encode categorical predictors (one-hot)
dataset_dummies <- fastDummies::dummy_cols(
  dataset,
  select_columns = cat_cols,
  remove_first_dummy = TRUE,      # avoid perfect collinearity
  remove_selected_columns = TRUE  # drop original categorical columns
)

# Step 12: Separate predictors and target again
features <- dataset_dummies %>% dplyr::select(-status)

# Step 13: Median imputation for remaining numeric NAs
for (col in names(features)) {
  if (is.numeric(features[[col]])) {
    if (any(is.na(features[[col]]))) {
      med <- median(features[[col]], na.rm = TRUE)
      if (is.na(med)) {
        features[[col]] <- 0
      } else {
        features[[col]][is.na(features[[col]])] <- med
      }
    }
  }
}

# Step 14: Scale all numeric predictors to [0,1]
scale_to_zero_one <- function(x) {
  if (!is.numeric(x)) return(x)
  if (all(is.na(x)))  return(rep(0, length(x)))
  mn <- min(x, na.rm = TRUE)
  mx <- max(x, na.rm = TRUE)
  if (mn == mx) return(rep(0, length(x)))
  scaled <- (x - mn) / (mx - mn)
  scaled[is.na(scaled)]       <- 0
  scaled[is.infinite(scaled)] <- 0
  scaled
}

features_scaled <- as.data.frame(lapply(features, scale_to_zero_one))
features_scaled_num <- as.data.frame(lapply(features_scaled, as.numeric))

# Final x and y
x <- data.matrix(features_scaled_num)
y_factor <- dataset_dummies$status
y_num <- as.numeric(y_factor) - 1
num_classes <- length(unique(y_num))



############################################################
# 2. Train / validation / test split (stratified)
############################################################


### Save Splits in csv files ###


set.seed(1)
train_index <- createDataPartition(y_num, p = 0.7, list = FALSE)

x_train <- x[train_index, ]
y_train <- y_num[train_index]

x_temp <- x[-train_index, ]
y_temp <- y_num[-train_index]

set.seed(1)
val_index <- createDataPartition(y_temp, p = 0.5, list = FALSE)

x_val <- x_temp[val_index, ]
y_val <- y_temp[val_index]

x_test <- x_temp[-val_index, ]
y_test <- y_temp[-val_index]

############################################################
# 3. Class weights to handle imbalance
############################################################

freq <- table(y_train)
raw_w <- 1 / sqrt(freq)      # softer than 1/freq
w <- raw_w / mean(raw_w)     # normalise around 1

class_weights <- as.list(as.numeric(w))
names(class_weights) <- names(freq)
print(class_weights)


############################################################
# 4. Hyperparameters via FLAGS (for tfruns::tuning_run)
############################################################

# Value here is only default value.for the experiment, the values are defined in the tuning script.

FLAGS <- flags(
  flag_numeric("learning_rate", 0.0001),
  flag_integer("batch_size",    256),
  flag_integer("units1",        512),
  flag_integer("units2",        256),
  flag_integer("units3",        128),      
  flag_integer("units4",        64), 
  flag_integer("units5",        32), 
  flag_integer("units6",        16), 
  
  flag_string ("act1",          "relu"),
  flag_string ("act2",          "relu"),
  flag_string ("act3",          "relu"),
  flag_string ("act4",          "relu"),
  flag_string ("act5",          "relu"),
  flag_string ("act6",          "relu"),
    
  flag_numeric("dropout1",       0.2),
  flag_numeric("dropout2",       0.2),
  flag_numeric("dropout3",       0.2),
  flag_numeric("dropout4",       0.2),
  flag_numeric("dropout5",       0.2),
  flag_numeric("dropout6",       0.2),
  flag_integer("epochs",        2000)
)

############################################################
# 5. Model definition (feed-forward network)
############################################################

l2_reg <- 0.002

model_ffn <- keras_model_sequential() %>%
  layer_dense(
    units            = FLAGS$units1,
    activation       = FLAGS$act1,
    input_shape      = ncol(x_train),
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(
    units            = FLAGS$units2,
    activation       = FLAGS$act2,
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(
    units            = FLAGS$units3,
    activation       = FLAGS$act3,
    kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(
    units            = FLAGS$units4,
    activation       = FLAGS$act4,
   kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout4) %>%
  layer_dense(
     units            = FLAGS$units5,
     activation       = FLAGS$act5,
  kernel_regularizer = regularizer_l2(l2_reg)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout5) %>%
  layer_dense(
     units            = FLAGS$units6,
     activation       = FLAGS$act6,
  kernel_regularizer = regularizer_l2(l2_reg)
  ) %>% 
  layer_batch_normalization() %>%
  layer_dropout(rate = FLAGS$dropout6) %>%
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
# 6. Training with early stopping + LR scheduling
############################################################

callback_es <- callback_early_stopping(
  monitor             = "val_loss",
  patience            = 100,
  restore_best_weights = TRUE
)

callback_lr <- callback_reduce_lr_on_plateau(
  monitor  = "val_loss",  # watch validation loss
  factor   = 0.5,         # multiply lr by 0.5 when plateau
  patience = 20,          # wait 10 epochs without improvement
  min_lr   = 1e-6,        # do not go below this learning rate
  verbose  = 1            # print when LR is reduced
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
# 7. Evaluation on test set (logged by tfruns)
############################################################

scores <- model_ffn %>% evaluate(
  x_test,
  y_test,
  verbose = 0
)

print(scores)

############################################################
# 8. Save model in this run's directory
############################################################

save_model(
  model_ffn,
  filepath  = file.path(run_dir(), "model.keras"),
  overwrite = TRUE
)

