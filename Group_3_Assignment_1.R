set.seed(1)

# ---------------------- Packages ----------------------
if(!require('tidyverse')) {             
  install.packages('tidyverse')
  library('tidyverse')
}
if(!require('lubridate')) {              
  install.packages('lubridate')
  library('lubridate')
}
if(!require('janitor')) {              
  install.packages('janitor')
  library('janitor')
}
if(!require('skimr')) {              
  install.packages('skimr')
  library('skimr')
}
if(!require('caret')) {              
  install.packages('caret')
  library('caret')
}
if(!require('glmnet')) {              
  install.packages('glmnet')
  library('glmnet')
}
if(!require('xgboost')) {              
  install.packages('xgboost')
  library('xgboost')
}
if(!require('ranger')) {              
  install.packages('ranger')
  library('ranger')
}

# ---------------------- Import data ----------------------
raw <- read_delim("LCdata.csv",
                  delim = ";",
                  na = c("", "NA","N/A"))

# ---------------------- Mark 200 "reality-check" observations ----------------------
# These 200 will NOT be used for train/test/model selection
set.seed(123)   # separate seed for reproducibility of the 200
raw <- raw %>% mutate(reality_flag = 0L)
idx_reality <- sample(nrow(raw), 1000)
raw$reality_flag[idx_reality] <- 1L

set.seed(1)

# ---------------------- Start preprocessing ----------------------
# Everything below works on ALL rows, but later we split by reality_flag

# Drop attributes not available for new applications
drop_now <- c(
  "collection_recovery_fee","installment","initial_list_status", "funded_amnt","funded_amnt_inv",
  "issue_d","last_pymnt_amnt","last_pymnt_d","loan_status","next_pymnt_d",
  "out_prncp","out_prncp_inv","pymnt_plan","recoveries","total_pymnt",
  "total_pymnt_inv","total_rec_int","total_rec_late_fee","total_rec_prncp"
)
dat <- raw %>% select(-any_of(drop_now))

# Drop IDs, URL and description
dat <- dat %>% select(-any_of(c("id", "member_id", "url", "desc")))

# Text / high-cardinality categoricals
dat <- dat %>% select(-any_of(c("emp_title", "title", "zip_code")))

# Create binary flags for missing derog/delinquency/record
dat <- dat %>%
  mutate(
    has_major_derog   = ifelse(is.na(mths_since_last_major_derog), 0, 1),
    has_delinq        = ifelse(is.na(mths_since_last_delinq), 0, 1),
    has_public_record = ifelse(is.na(mths_since_last_record), 0, 1)
  ) %>%
  select(-mths_since_last_major_derog,
         -mths_since_last_delinq,
         -mths_since_last_record)

# Drop derived/redundant variables, sub_grade is determined by grade therefore if you know grad information in sub_grade is redundant
# also policy code is always 1 meaning no information for prediction
dat <- dat %>% select(-any_of(c("sub_grade", "policy_code")))

# Joint applications
dat <- dat %>%
  mutate(is_joint = ifelse(application_type == "JOINT", 1, 0)) %>%
  select(-annual_inc_joint, -dti_joint, -verification_status_joint)
dat <- dat %>% select(-is_joint)

# Remove “block dependency” columns (optional credit bureau feed)
block_vars <- c("open_acc_6m","open_il_6m","open_il_12m","open_il_24m",
                "mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m",
                "open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m")
dat <- dat %>% select(-any_of(block_vars))

# Remove block dependency rows (lack of credit history)
block_25 <- c("delinq_2yrs","inq_last_6mths","open_acc",
              "pub_rec","total_acc","acc_now_delinq")
dat <- dat %>% filter(!if_any(all_of(block_25), is.na))

# Block dependency rows II (credit summary)
block_63k <- c("tot_coll_amt","tot_cur_bal","total_rev_hi_lim")
dat <- dat %>%
  mutate(has_credit_summary = ifelse(is.na(tot_coll_amt), 0, 1)) %>%
  mutate(across(all_of(block_63k),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Replace NAs in revol_bal with median
dat <- dat %>%
  mutate(revol_bal = ifelse(is.na(revol_bal),
                            median(revol_bal, na.rm = TRUE),
                            revol_bal))

# Replace NAs in revol_util with 0
dat <- dat %>%
  mutate(revol_util = ifelse(is.na(revol_util), 0, revol_util))

# Replace NAs in collections_12_mths_ex_med with 0
dat <- dat %>%
  mutate(collections_12_mths_ex_med = ifelse(is.na(collections_12_mths_ex_med),
                                             0,
                                             collections_12_mths_ex_med))

# Convert selected categoricals to factors
factor_cols <- c("term","grade","emp_length","home_ownership","verification_status",
                 "purpose","addr_state","initial_list_status","application_type")

factor_cols <- intersect(factor_cols, names(dat))
dat <- dat %>% mutate(across(all_of(factor_cols), as.factor))

# Drop last_credit_pull_d
dat <- dat %>% select(-last_credit_pull_d)

# Date handling: earliest_cr_line -> age in years
library(stringr)
dat$earliest_cr_line <- my(dat$earliest_cr_line)
dat <- dat %>%
  mutate(
    earliest_cr_line_age_years =
      time_length(interval(earliest_cr_line, today()), "years")
  ) %>%
  select(-earliest_cr_line)

# Employment flag
dat <- dat %>%
  mutate(emp_length = na_if(str_to_lower(emp_length), "n/a")) %>%
  mutate(
    employed_flag = ifelse(is.na(emp_length), 0, 1)
  ) %>%
  select(-emp_length)

# Home ownership grouping + dummies
dat <- dat %>%
  mutate(
    home_ownership = case_when(
      home_ownership %in% c("ANY", "NONE", "OTHER") ~ "OTHER",
      TRUE ~ as.character(home_ownership)
    )
  ) %>%
  mutate(home_ownership = as.factor(home_ownership))

dummies_home <- dummyVars(" ~ home_ownership", data = dat)
home_dummies <- predict(dummies_home, newdata = dat)
dat <- cbind(dat, home_dummies) %>%
  select(-home_ownership)

# Purpose grouping + dummies
dat <- dat %>%
  mutate(
    purpose = case_when(
      purpose %in% c("debt_consolidation", "credit_card",
                     "home_improvement", "other") ~ as.character(purpose),
      TRUE ~ "other"
    ),
    purpose = as.factor(purpose)
  )

dummies_purpose <- dummyVars(" ~ purpose", data = dat)
purpose_dummies <- predict(dummies_purpose, newdata = dat)
dat <- cbind(dat, purpose_dummies) %>%
  select(-purpose)

# ----- Outliers and transformations -----

# Annual income: cut extremes and log-transform
q_low  <- quantile(dat$annual_inc, 0.01, na.rm = TRUE)
q_high <- quantile(dat$annual_inc, 0.99, na.rm = TRUE)

outliers <- sum(dat$annual_inc < q_low |
                  dat$annual_inc > q_high |
                  dat$annual_inc == 0, na.rm = TRUE)
outlier_percent <- (outliers / nrow(dat)) * 100

cat("Outlier threshold low:", q_low, "\n")
cat("Outlier threshold high:", q_high, "\n")
cat("Number of outliers:", outliers, "\n")
cat("Percentage of outliers:", round(outlier_percent, 2), "%\n")

dat <- dat %>%
  filter(annual_inc > 0 & annual_inc <= q_high) %>%
  mutate(annual_inc_log = log1p(annual_inc)) %>%
  select(-annual_inc)

# Cut DTI, delinq_2yrs, cap inq_last_6mths, consistency between open_acc and total_acc
dat <- dat %>% filter(dti <= 100)
dat <- dat %>% filter(delinq_2yrs <= 10)
dat <- dat %>% mutate(inq_last_6mths = pmin(inq_last_6mths, 6))
dat <- dat %>% filter(open_acc <= total_acc)

# pub_rec -> binary flag
dat <- dat %>%
  mutate(pub_rec_flag = ifelse(pub_rec > 0, 1, 0)) %>%
  select(-pub_rec)

# Verification status dummies
dummies_verif <- dummyVars(" ~ verification_status", data = dat)
verification_dummies <- predict(dummies_verif, newdata = dat)
dat <- cbind(dat, verification_dummies) %>%
  select(-verification_status)

# Drop baseline dummies to avoid perfect multicollinearity
dat <- dat %>%
  select(
    -any_of(c("verification_status.Not Verified",
              "purpose.other",
              "home_ownership.RENT"))
  )

# ---------------------- Split out reality set ----------------------
# At this point, dat still contains 'reality_flag' from raw

# Keep reality rows separate
reality_dat <- dat %>%
  filter(reality_flag == 1) %>%
  select(-reality_flag)

# Keep modeling rows (used for train/test and model selection)
dat_model <- dat %>%
  filter(reality_flag == 0) %>%
  select(-reality_flag)

# ---------------------- Train/test split on modeling data ----------------------
dat <- as.data.frame(dat_model)

# Make names safe for model formula / xgboost / ranger
names(dat)        <- make.names(names(dat))
names(reality_dat) <- make.names(names(reality_dat))

# sanity checks
stopifnot(is.data.frame(dat))
stopifnot(!any(sapply(dat, is.character)))
stopifnot(sum(is.na(dat$int_rate)) == 0)

# Train/test split
set.seed(1)
idx <- createDataPartition(dat$int_rate, p = 0.8, list = FALSE)
train <- dat[idx, , drop = FALSE]
test  <- dat[-idx, , drop = FALSE]

# Cross-validation control
k <- 5
ctrl <- trainControl(
  method = "cv",
  number = k,
  verboseIter = FALSE
)

# ---------------------- Linear regression ----------------------
model_lm <- lm(int_rate ~ ., data = train)
summary(model_lm)

pred_train_lm <- predict(model_lm, newdata = train)
mse_train_lm  <- mean((train$int_rate - pred_train_lm)^2)

pred_test_lm  <- predict(model_lm, newdata = test)
mse_lm        <- mean((test$int_rate - pred_test_lm)^2)
rmse_lm       <- sqrt(mse_lm)

# CV on LM
cv_lm <- train(
  int_rate ~ .,
  data = train,
  method = "lm",
  trControl = ctrl,
  metric = "RMSE"
)
cv_rmse_lm <- cv_lm$results$RMSE[1]
cv_mse_lm  <- cv_rmse_lm^2

# ---------------------- Lasso regression (glmnet) ----------------------
x_train <- model.matrix(int_rate ~ ., data = train)[, -1]
y_train <- train$int_rate

# cross-validated lasso
cv <- cv.glmnet(x_train, y_train, alpha = 1)  # Lasso

# Predictions on test
x_test <- model.matrix(int_rate ~ ., data = test)[, -1]
pred_lasso <- predict(cv, s = "lambda.min", newx = x_test)
mse_lasso  <- mean((test$int_rate - as.numeric(pred_lasso))^2)
rmse_lasso <- sqrt(mse_lasso)

# CV on Lasso via caret (optional, for comparison)
cv_lasso <- train(
  int_rate ~ .,
  data = train,
  method = "glmnet",
  trControl = ctrl,
  tuneLength = 10,
  metric = "RMSE"
)
best_row_lasso <- cv_lasso$results[which.min(cv_lasso$results$RMSE), ]
cv_rmse_lasso  <- best_row_lasso$RMSE
cv_mse_lasso   <- cv_rmse_lasso^2

# ---------------------- XGBoost ----------------------
X_train <- model.matrix(int_rate ~ ., data = train)[, -1]
y_train <- train$int_rate

xgb <- xgboost(
  data = X_train,
  label = y_train,
  nrounds = 300,
  max_depth = 6,
  eta = 0.05,
  objective = "reg:squarederror",
  verbose = 0
)

X_test <- model.matrix(int_rate ~ ., data = test)[, -1]
pred_xgb <- predict(xgb, newdata = X_test)
mse_xgb  <- mean((test$int_rate - pred_xgb)^2)
rmse_xgb <- sqrt(mse_xgb)

# CV on XGBoost via caret
cv_xgb <- train(
  int_rate ~ .,
  data = train,
  method = "xgbTree",
  trControl = ctrl,
  tuneLength = 3,
  metric = "RMSE"
)
best_row_xgb <- cv_xgb$results[which.min(cv_xgb$results$RMSE), ]
cv_rmse_xgb  <- best_row_xgb$RMSE
cv_mse_xgb   <- cv_rmse_xgb^2

# ---------------------- Random Forest (ranger) ----------------------
set.seed(1)
rf_model <- ranger(
  int_rate ~ .,
  data = train,
  num.trees = 300,
  mtry = floor(sqrt(ncol(train) - 1)),
  min.node.size = 5,
  importance = "impurity",
  respect.unordered.factors = "order",
  seed = 1
)

# OOB error (MSE) from ranger
oob_mse_rf  <- rf_model$prediction.error
oob_rmse_rf <- sqrt(oob_mse_rf)

# Predict on test
rf_pred <- predict(rf_model, data = test)$predictions
mse_rf  <- mean((test$int_rate - rf_pred)^2)
rmse_rf <- sqrt(mse_rf)

mae_rf  <- mean(abs(test$int_rate - rf_pred))
ss_tot  <- sum((test$int_rate - mean(test$int_rate))^2)
ss_res  <- sum((test$int_rate - rf_pred)^2)
r2_rf   <- 1 - ss_res / ss_tot

cat("RF RMSE (test):", round(rmse_rf, 3), "\n")
cat("RF MAE  (test):", round(mae_rf, 3), "\n")
cat("RF R2   (test):", round(r2_rf, 3), "\n")
cat("RF OOB RMSE   :", round(oob_rmse_rf, 3), "\n")


# ---------------------- Combined results table ----------------------

results_all <- data.frame(
  Model = c("Linear regression",
            "Lasso regression",
            "XGBoost",
            "Random Forest"),
  
  Train_MSE = c(mse_train_lm,
                mse_train_lasso,
                mse_train_xgb,
                mse_train_rf),
  
  Test_MSE = c(mse_lm,
               mse_lasso,
               mse_xgb,
               mse_rf),
  
  CV_like_MSE = c(cv_mse_lm,
                  cv_mse_lasso,
                  cv_mse_xgb,
                  oob_mse_rf)
)

results_all$Train_MSE   <- round(results_all$Train_MSE, 3)
results_all$Test_MSE    <- round(results_all$Test_MSE, 3)
results_all$CV_like_MSE <- round(results_all$CV_like_MSE, 3)

print(results_all)


# ---------------------- Reality-check evaluation ----------------------


# --- Example for XGBoost as best model ---
X_reality_xgb <- model.matrix(int_rate ~ ., data = reality_dat)[, -1]
pred_reality_xgb <- predict(xgb, newdata = X_reality_xgb)
mse_reality_xgb  <- mean((reality_dat$int_rate - pred_reality_xgb)^2)
rmse_reality_xgb <- sqrt(mse_reality_xgb)

cat("XGB - Reality-check RMSE:", round(rmse_reality_xgb, 3), "\n")
cat("XGB - Reality-check MSE :", round(mse_reality_xgb, 3), "\n")
