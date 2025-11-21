# Seed for reproducibility
set.seed(1)

# Load required packages
if(!require('tidyverse')) {             
  install.packages('tidyverse')
  library('tidyverse')
}

if(!require('caret')) {             
  install.packages('caret')
  library('caret')
}

if(!require('janitor')) {             
  install.packages('janitor')
  library('janitor')
}

if(!require('skimr')) {             
  install.packages('skimr')
  library('skimr')
}

if(!require('keras3')) {
  install.packages('keras3')
  library('keras3')
}

if(!require('tensorflow')) {
  install.packages('tensorflow')
  library('tensorflow')
}

# tensorflow::install_tensorflow() 
# Run the line above once manually if tensorflow backend is not installed


############################################
# Data loading and preparation
############################################

# Step 1 Load raw data and remove technical ID column
# What  Read the original data from csv and drop ID which does not contain predictive information
raw <- read.csv("Dataset-part-2.csv", stringsAsFactors = FALSE)
if("ID" %in% names(raw)) {
  dataset <- raw %>% dplyr::select(-ID)
}

# Step 2 Clean character columns by replacing empty strings with NA
# What  Replace empty text cells with proper NA so that missing value handling works correctly
for(col in names(dataset)) {
  if(is.character(dataset[[col]])) {
    dataset[[col]][dataset[[col]] == ""] <- NA
  }
}

# Step 3 Convert target column status to factor
# What  Status is the class label and must be a factor before encoding and modeling
dataset$status <- as.factor(dataset$status)

# Step 4 Handle special placeholder values in DAYS_EMPLOYED
# What  Value 365243 indicates unemployed  create a flag and replace placeholder with NA
dataset$is_unemployed <- ifelse(dataset$DAYS_EMPLOYED == 365243, 1, 0)
dataset$DAYS_EMPLOYED[dataset$DAYS_EMPLOYED == 365243] <- NA

# Step 5 Create more interpretable age and employment duration features
# What  Convert days to years for better scale and interpretation
dataset$AGE <- abs(dataset$DAYS_BIRTH) / 365
dataset$YEARS_EMPLOYED <- abs(dataset$DAYS_EMPLOYED) / 365

# Optionally remove original day based columns to avoid redundancy
dataset <- dataset %>% dplyr::select(-DAYS_BIRTH, -DAYS_EMPLOYED)

# Step 6 Treat missing values in OCCUPATION_TYPE as a separate category
# What  NA in occupation often means no formal occupation  this is informative and should be kept as own group
dataset$OCCUPATION_TYPE[is.na(dataset$OCCUPATION_TYPE)] <- "Unknown"

# Step 7 Cap extreme values in CNT_FAM_MEMBERS and CHilDREN
# What  Limit unrealistic high family sizes to reduce influence of outliers
dataset$CNT_FAM_MEMBERS <- pmin(dataset$CNT_FAM_MEMBERS, 10)
dataset$CNT_CHILDREN <- pmin(dataset$CNT_CHILDREN, 6)


# Step 8 Apply log transform to income to reduce skewness
# What  Income distribution is heavy tailed  log transform makes it more suitable for neural networks
dataset$AMT_INCOME_TOTAL <- log1p(dataset$AMT_INCOME_TOTAL)


# Step 11 Remove uninformative constant feature FLAG_MOBIL
# What  FLAG_MOBIL is always one in this dataset and does not help prediction
if("FLAG_MOBIL" %in% names(dataset)) {
  dataset <- dataset %>% dplyr::select(-FLAG_MOBIL)
}

# Step 12 Separate predictors and target
# What  Split data into features for input and status as output label
features <- dataset %>% dplyr::select(-status)

# Step 13 Encode character predictors as numeric codes
# What  Neural networks require numeric input  convert each category to an integer code
for(col in names(features)) {
  if(is.character(features[[col]])) {
    features[[col]] <- as.numeric(as.factor(features[[col]]))
  }
}

# Step 14 Impute remaining numeric missing values with median
# What  Neural networks cannot handle NA  use simple and robust median imputation per column
for(col in names(features)) {
  if(is.numeric(features[[col]])) {
    if(any(is.na(features[[col]]))) {
      med <- median(features[[col]], na.rm = TRUE)
      if(is.na(med)) {
        features[[col]] <- 0
      } else {
        features[[col]][is.na(features[[col]])] <- med
      }
    }
  }
}

# Step 15 Scale all numeric predictors to range zero to one
# What  Scaling stabilizes training of neural networks and makes features comparable
scale_to_zero_one <- function(x) {
  if(is.numeric(x)) {
    if(all(is.na(x))) {
      return(rep(0, length(x)))
    }
    mn <- min(x, na.rm = TRUE)
    mx <- max(x, na.rm = TRUE)
    if(mn == mx) {
      return(rep(0, length(x)))
    }
    scaled <- (x - mn) / (mx - mn)
    scaled[is.na(scaled)] <- 0
    scaled[is.infinite(scaled)] <- 0
    return(scaled)
  } else {
    return(x)
  }
}

features_scaled <- as.data.frame(lapply(features, scale_to_zero_one))

# Ensure numeric type for all predictors
features_scaled_num <- as.data.frame(lapply(features_scaled, function(col) {
  as.numeric(col)
}))

# Step 16 Create final input matrix x and numeric target vector y
# What  Prepare data in matrix form with integer class labels for keras sparse categorical loss
x <- data.matrix(features_scaled_num)
y_factor <- dataset$status
y_num <- as.numeric(y_factor) - 1

# Optional sanity checks
summary(x)
any(is.na(x))
table(y_num)

# Step 17 Create train validation test split
# What  Split the data to evaluate training performance and generalization

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

# Sanity check sizes
dim(x_train); length(y_train)
dim(x_val); length(y_val)
dim(x_test); length(y_test)

############################################
# Simple neural network as starting point
############################################

# Step 19 Determine number of classes for softmax output
# What  Count unique class labels in y for the output layer size
num_classes <- length(unique(y_num))

# Step 20 Define a simple baseline neural network model
# What  One hidden layer with relu and softmax output for multi class classification

model_simple <- keras_model_sequential() %>%
  layer_dense(
    units = 32,
    activation = "relu",
    input_shape = ncol(x_train)
  ) %>%
  layer_dense(
    units = num_classes,
    activation = "softmax"
  )

# Step 21 Compile the model
# What  Use Adam optimizer and sparse categorical crossentropy for integer class labels

model_simple %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

# Optional show model summary
summary(model_simple)

# Step 22 Train the simple model
# What  Fit the model on training data and monitor validation accuracy

history_simple <- model_simple %>% fit(
  x_train,
  y_train,
  epochs = 30,
  batch_size = 128,
  validation_data = list(x_val, y_val),
  verbose = 1
)

# Optional plot training history
plot(history_simple)

# Step 23 Evaluate the model on test data
# What  Test set accuracy is used to judge baseline model quality

test_results_simple <- model_simple %>% evaluate(
  x_test,
  y_test,
  verbose = 0
)

print(test_results_simple)

# Step 24 Save the trained simple model for later comparison and reality check
save_model(
  model_simple,
  overwrite = TRUE, 
  "nn_simple_assignment2.keras"
)

############################################
# Improved neural network model
# Uses deeper architecture, batch normalization and dropout
############################################

# Determine number of classes in case not yet defined
num_classes <- length(unique(y_num))

# Define improved model architecture
# What  Three hidden layers with relu activations, batch normalization and dropout
model_improved <- keras_model_sequential() %>%
  layer_dense(
    units = 128,
    activation = "relu",
    input_shape = ncol(x_train)
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_dense(
    units = 64,
    activation = "relu"
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  
  layer_dense(
    units = 32,
    activation = "relu"
  ) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.1) %>%
  
  layer_dense(
    units = num_classes,
    activation = "softmax"
  )

# Compile improved model
# What  Use Adam with a moderate learning rate and sparse categorical crossentropy
model_improved %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss      = "sparse_categorical_crossentropy",
  metrics   = "accuracy"
)

# Show model summary
summary(model_improved)

# Define callbacks
# What  Early stopping prevents overfitting, ReduceLROnPlateau lowers learning rate when validation loss stops improving
cb_early <- callback_early_stopping(
  monitor           = "val_loss",
  patience          = 6,
  restore_best_weights = TRUE
)

cb_plateau <- callback_reduce_lr_on_plateau(
  monitor  = "val_loss",
  factor   = 0.5,
  patience = 3,
  min_lr   = 1e-5
)

# Train improved model
# What  Train on training data and monitor performance on validation data
history_improved <- model_improved %>% fit(
  x_train,
  y_train,
  epochs          = 80,
  batch_size      = 128,
  validation_data = list(x_val, y_val),
  callbacks       = list(cb_early, cb_plateau),
  verbose         = 1
)

# Optional plot of training history
plot(history_improved)

# Evaluate improved model on test data
# What  Test accuracy of improved model for comparison with simple model
test_results_improved <- model_improved %>% evaluate(
  x_test,
  y_test,
  verbose = 0
)

print(test_results_improved)

# Save improved model
save_model(
  model_improved,
  overwrite = TRUE, 
  "nn_improved_assignment2.keras"
)


