set.seed(1)

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

# Import CSV with semicolon as eliminator/delimiter
raw <- read_delim("LCdata.csv",
                  delim = ";",       # <-- semicolon is the eliminator
                  na = c("", "NA","N/A"))  # handle "", "NA" and "N/A" as missing values


# Drop attributes not available for new applications, because they won’t exist at prediction time
drop_now <- c(
  "collection_recovery_fee","installment","initial_list_status", "funded_amnt","funded_amnt_inv",
  "issue_d","last_pymnt_amnt","last_pymnt_d","loan_status","next_pymnt_d",
  "out_prncp","out_prncp_inv","pymnt_plan","recoveries","total_pymnt",
  "total_pymnt_inv","total_rec_int","total_rec_late_fee","total_rec_prncp"
)
dat <- raw %>% select(-any_of(drop_now))


# ----- Drop IDs, URL and description which are not relevant for predictions -----
dat <- dat %>% select(-any_of(c("id", "member_id", "url", "desc")))


# ----- Text or high-cardinality categorical variables -----
dat <- dat %>% select(-any_of(c("emp_title", "title", "zip_code")))

# emp_title = “Teacher”, “Software Engineer”, “CEO” → too specific.
# title = self-assigned loan purpose, duplicates “purpose”.
# zip_code = very granular; state-level (addr_state) already captures region.

# ----- Text or high-cardinality categorical variables -----
dat <- dat %>%
  mutate(
    has_major_derog = ifelse(is.na(mths_since_last_major_derog), 0, 1),
    has_delinq = ifelse(is.na(mths_since_last_delinq), 0, 1),
    has_public_record = ifelse(is.na(mths_since_last_record), 0, 1)
  ) %>%
  select(-mths_since_last_major_derog,
         -mths_since_last_delinq,
         -mths_since_last_record)

# mths_since_last_major_derog:	Months since last major derogatory remark 	~75% missing	Drop numeric, add binary has_major_derog
# mths_since_last_delinq:     	Months since last delinquency             	~51% missing	Drop numeric, add binary has_delinq
# mths_since_last_record:     	Months since last public record           	~85% missing	Drop numeric, add binary has_public_record

# we assume that data collection is complet and here missing values are not errors; they indicate no record.

# ----- Derived or redundant variables -----
dat <- dat %>% select(-any_of(c("sub_grade", "policy_code")))

# sub_grade encodes finer levels of grade.→ Keep only one. Usually keep grade, drop sub_grade.
# policy_code has constant value (usually “1”) → drop.


# ----- Redundant variables, joint applications -----
dat <- dat %>%
  mutate(is_joint = ifelse(application_type == "JOINT", 1, 0)) %>%
  select(-annual_inc_joint, -dti_joint, -verification_status_joint)
dat <- dat %>% select(-is_joint)

# keep information of joint application or not in the data
# remove the variables with a lot of missing values (all joints) 


# ----- Remove block dependency columns  -----
# In Lending Club data, these “recent credit metrics” were only reported for loans that have a credit bureau file with a specific version of the record.
# Older or differently sourced loan applications don’t contain them at all — therefore, these NAs are not “missing at random”, but missing by design.
# come from the same optional credit bureau feed
# Optional: add a simple binary flag indicating whether this block exists for an applicant

block_vars <- c("open_acc_6m","open_il_6m","open_il_12m","open_il_24m",
                "mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m",
                "open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m")

dat <- dat %>% select(-any_of(block_vars))


# ----- Remove block dependency rows (lack of credit history) -----
# Those six variables depend on each other — they are all missing together for 25 borrowers who lack credit-history data.

block_25 <- c("delinq_2yrs","inq_last_6mths","open_acc",
              "pub_rec","total_acc","acc_now_delinq")

dat <- dat %>% filter(!if_any(all_of(block_25), is.na))


# ----- Remove block dependency rows II (lack of credit history) -----
# These three columns come from the same credit-bureau record set.
# When Lending Club pulled the credit data, older loans and some thin-file applicants did not yet have these derived totals recorded in the database.
# Because ~92 % of rows have data, they still carry predictive information. add a flag and replace Nas

 # Group the three correlated variables 
block_63k <- c("tot_coll_amt","tot_cur_bal","total_rev_hi_lim")

# add flag for missing credit summary (for 8% of the sample)
dat <- dat %>%
  mutate(has_credit_summary = ifelse(is.na(tot_coll_amt), 0, 1))

# Replace NAs with Median
dat <- dat %>%
  mutate(across(all_of(block_63k),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))


# ----- Replace NAs with Median in revol_bal -----
# only two rows out of the sample, randomly missing with low impact on Model. Just replace with Median.

dat <- dat %>%
  mutate(revol_bal = ifelse(is.na(revol_bal), median(revol_bal, na.rm = TRUE), revol_bal))


# ----- Replace NAs with 0 in revol_util -----
# the amount of credit the borrower is using relative to all available revolving credit lines.
# These missing values usually occur for borrowers with no revolving credit lines. If the borrower has no revolving credit, utilization = 0% is a sensible assumption.
# So the missingness isn’t random — it’s logical, or missing by design.

dat <- dat %>%
  mutate(revol_util = ifelse(is.na(revol_util), 0, revol_util))


# ----- Replace NAs with 0 in collections_12_mths_ex_med -----
# It counts how many times the borrower’s accounts were sent to collections (non-medical) in the last year.
# 0 → no collection events → typical for most borrowers
# 1 or more → indicates financial distress
dat <- dat %>%
  mutate(collections_12_mths_ex_med = ifelse(is.na(collections_12_mths_ex_med), 0, collections_12_mths_ex_med))


# ----- Convert categorical variables to factors -----
factor_cols <- c("term","grade","emp_length","home_ownership","verification_status",
                 "purpose","addr_state","initial_list_status","application_type")

# This line makes sure you don’t try to convert a column that doesn’t exist in your dataset.
# names(dat) = all column names currently in your data.
# intersect() = takes only the names that appear in both vectors.

factor_cols <- intersect(factor_cols, names(dat))
dat <- dat %>% mutate(across(all_of(factor_cols), as.factor))



# ----- drop beacuase they have to check anyway when applying -------
dat <- dat %>% select(-last_credit_pull_d)


# -----  Calculate difference (in years) between today and last credit pull
library(dplyr)
library(stringr)
library(lubridate)


dat$earliest_cr_line <- my(dat$earliest_cr_line)

# Compute YEARS since earliest credit line (numeric)
dat <- dat %>%
  mutate(
    earliest_cr_line_age_years = time_length(interval(earliest_cr_line, today()), "years")
  ) %>%
  select( -earliest_cr_line)  # tidy up helpers


# ------ create binary flag for epmployeed or not ---------

dat <- dat %>%
  mutate(emp_length = na_if(str_to_lower(emp_length), "n/a"))
dat <- dat %>%
  mutate(
    employed_flag = ifelse(is.na(emp_length), 0, 1)
  ) %>%
  select(-emp_length)


# ----- create dummy variables and group for home_ownership ----
dat <- dat %>%
  mutate(
    home_ownership = case_when(
      home_ownership %in% c("ANY", "NONE", "OTHER") ~ "OTHER",
      TRUE ~ home_ownership
    )
  )

dat <- dat %>%
  mutate(home_ownership = as.factor(home_ownership))


dummies <- dummyVars(" ~ home_ownership", data = dat)
home_dummies <- predict(dummies, newdata = dat)
dat <- cbind(dat, home_dummies)

dat <- dat %>% select(-any_of("home_ownership"))


# ---- dummy variable purpose ---
dat <- dat %>%
  mutate(
    purpose = case_when(
      purpose %in% c("debt_consolidation", "credit_card", "home_improvement", "other") ~ purpose,
      TRUE ~ "other"   # all other categories merged into "other"
    ),
    purpose = as.factor(purpose)
  )

# Create dummy variables for the purpose variable
dummies <- dummyVars(" ~ purpose", data = dat)
purpose_dummies <- predict(dummies, newdata = dat)

# Add dummy columns to your dataset
dat <- cbind(dat, purpose_dummies)

dat <- dat %>% select(-any_of("purpose"))



#----- outliers--------------  maybe check with other parameters


# Compute thresholds
q_low  <- quantile(dat$annual_inc, 0.01, na.rm = TRUE)
q_high <- quantile(dat$annual_inc, 0.99, na.rm = TRUE)

# Count outliers (below 1st percentile, above 99th percentile, or 0 income)
outliers <- sum(dat$annual_inc < q_low | dat$annual_inc > q_high | dat$annual_inc == 0, na.rm = TRUE)

# Compute percentage
outlier_percent <- (outliers / nrow(dat)) * 100

# Print results
cat("Outlier threshold low:", q_low, "\n")
cat("Outlier threshold high:", q_high, "\n")
cat("Number of outliers:", outliers, "\n")
cat("Percentage of outliers:", round(outlier_percent, 2), "%\n")

dat <- dat %>%
  filter(annual_inc > 0 & annual_inc <= q_high)

# Use log income (often more linear):
dat <- dat %>% mutate(annual_inc_log = log1p(annual_inc)) %>% select(-annual_inc)


# cut everything above 100% dti
dat <- dat |> filter(dti <= 100)
# cut delinq_2yrs at 10
dat <- dat |> filter(delinq_2yrs <= 10)

# cap inq_last_6mths at 6
dat <- dat |> mutate(inq_last_6mths = pmin(inq_last_6mths, 6))
#remove errors where open_account > total_acc
dat <- dat |> filter(open_acc <= total_acc)

# transform pub_rec into binary variable, keep binary variable and remove the origin variable
dat <- dat %>%
  mutate(pub_rec_flag = ifelse(pub_rec > 0, 1, 0))
dat <- dat %>% select(-any_of("pub_rec"))


# Create dummy variables for verification_status, instead of order them. so we can use it better in tree-models
dummies <- dummyVars(" ~ verification_status", data = dat)
verification_dummies <- predict(dummies, newdata = dat)

# Add dummy columns to your dataset and remove origin variable
dat <- cbind(dat, verification_dummies)
dat <- dat %>% select(-any_of("verification_status"))

# drop one dummy variable for each group to avoid perfect multicollinearity. the dropped ones ar then the reference category
dat <- dat %>%
# choose baselines: Not.Verified, other, RENT (adjust if needed)
select(
    -any_of(c("verification_status.Not Verified",
              "purpose.other",
              "home_ownership.RENT"))
  )



##################### train & evaluate models ########################

set.seed(1)   # reproducible split, ok for assignment

# Make sure dat is a data.frame and target present
dat <- as.data.frame(dat)
stopifnot("int_rate" %in% names(dat))
stopifnot(sum(is.na(dat$int_rate)) == 0)

# Helper functions
mse  <- function(y, yhat) mean((y - yhat)^2)
rmse <- function(y, yhat) sqrt(mse(y, yhat))

#---------------- 1) Validation Set Approach (train / test split) ---------------

# use same style as cv_vsa lab: sample() on row indices
n <- nrow(dat)
train_idx <- sample(n, floor(0.8 * n))  # 80% train, 20% test

train <- dat[train_idx, ]
test  <- dat[-train_idx, ]

y_train <- train$int_rate
y_test  <- test$int_rate

#---------------- 2) Model 1: Ordinary Least Squares (lm) -----------------------

# full linear model with all prepared predictors
lm_fit <- lm(int_rate ~ ., data = train)

# Look at model (for interpretation in report)
summary(lm_fit)

# Training performance
pred_train_lm <- predict(lm_fit, newdata = train)
train_mse_lm  <- mse(y_train, pred_train_lm)
train_rmse_lm <- rmse(y_train, pred_train_lm)

# Test performance
pred_test_lm <- predict(lm_fit, newdata = test)
test_mse_lm  <- mse(y_test, pred_test_lm)
test_rmse_lm <- rmse(y_test, pred_test_lm)

cat("Linear regression (OLS)\n")
cat("  Train MSE :", round(train_mse_lm, 4),
    "RMSE:", round(train_rmse_lm, 4), "\n")
cat("  Test  MSE :", round(test_mse_lm, 4),
    "RMSE:", round(test_rmse_lm, 4), "\n\n")


#---------------- 3) k-fold Cross-Validation for OLS (cv.glm) -------------------

# follow cv_vsa lab: glm() + cv.glm()
if (!require("boot")) {
  install.packages("boot")
  library(boot)
} else {
  library(boot)
}

# glm with same formula as lm (fit on full data, cv.glm will re-split)
glm_ols <- glm(int_rate ~ ., data = dat)

# e.g. 5-fold CV (you can also try K=10)
set.seed(1)
cv_out_ols <- cv.glm(dat, glm_ols, K = 5)

# cv.glm() returns delta: [1]=raw CV MSE, [2]=adjusted CV MSE
cv_mse_ols  <- cv_out_ols$delta[2]
cv_rmse_ols <- sqrt(cv_mse_ols)

cat("OLS (glm) k-fold CV:\n")
cat("  CV MSE  :", round(cv_mse_ols, 4),
    "RMSE:", round(cv_rmse_ols, 4), "\n\n")


#---------------- 4) Model 2: Ridge & 3: LASSO (glmnet) -------------------------

if (!require("glmnet")) {
  install.packages("glmnet")
  library(glmnet)
} else {
  library(glmnet)
}

# model.matrix like in ridge/lasso lab: remove intercept column
X_train <- model.matrix(int_rate ~ ., data = train)[, -1]
X_test  <- model.matrix(int_rate ~ ., data = test)[, -1]

#---- Ridge (alpha = 0) ----
set.seed(1)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)  # default 10-fold CV

lambda_ridge <- cv_ridge$lambda.min

# training & test predictions at best lambda
pred_train_ridge <- predict(cv_ridge, s = lambda_ridge, newx = X_train)
pred_test_ridge  <- predict(cv_ridge, s = lambda_ridge, newx = X_test)

train_mse_ridge  <- mse(y_train, pred_train_ridge)
train_rmse_ridge <- rmse(y_train, pred_train_ridge)
test_mse_ridge   <- mse(y_test,  pred_test_ridge)
test_rmse_ridge  <- rmse(y_test,  pred_test_ridge)

# CV-MSE is stored in cv_ridge$cvm (vector over lambdas)
idx_ridge <- which(cv_ridge$lambda == lambda_ridge)
cv_mse_ridge  <- cv_ridge$cvm[idx_ridge]
cv_rmse_ridge <- sqrt(cv_mse_ridge)

cat("Ridge regression (alpha = 0)\n")
cat("  Best lambda:", signif(lambda_ridge, 3), "\n")
cat("  Train RMSE :", round(train_rmse_ridge, 4), "\n")
cat("  Test  RMSE :", round(test_rmse_ridge, 4), "\n")
cat("  CV    RMSE :", round(cv_rmse_ridge, 4), "\n\n")


#---- LASSO (alpha = 1) ----
set.seed(1)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)

lambda_lasso <- cv_lasso$lambda.min

pred_train_lasso <- predict(cv_lasso, s = lambda_lasso, newx = X_train)
pred_test_lasso  <- predict(cv_lasso, s = lambda_lasso, newx = X_test)

train_mse_lasso  <- mse(y_train, pred_train_lasso)
train_rmse_lasso <- rmse(y_train, pred_train_lasso)
test_mse_lasso   <- mse(y_test,  pred_test_lasso)
test_rmse_lasso  <- rmse(y_test,  pred_test_lasso)

idx_lasso <- which(cv_lasso$lambda == lambda_lasso)
cv_mse_lasso  <- cv_lasso$cvm[idx_lasso]
cv_rmse_lasso <- sqrt(cv_mse_lasso)

cat("LASSO regression (alpha = 1)\n")
cat("  Best lambda:", signif(lambda_lasso, 3), "\n")
cat("  Train RMSE :", round(train_rmse_lasso, 4), "\n")
cat("  Test  RMSE :", round(test_rmse_lasso, 4), "\n")
cat("  CV    RMSE :", round(cv_rmse_lasso, 4), "\n\n")


#---------------- 5) Put all errors into one comparison table -------------------

error_matrix <- data.frame(
  Model   = c("OLS (lm)",   "Ridge",           "LASSO"),
  TrainRMSE = c(train_rmse_lm, train_rmse_ridge, train_rmse_lasso),
  TestRMSE  = c(test_rmse_lm,  test_rmse_ridge,  test_rmse_lasso),
  CV_RMSE   = c(cv_rmse_ols,   cv_rmse_ridge,    cv_rmse_lasso)
)

print(error_matrix)

# From this table you can:
# - decide which model has the lowest TestRMSE (VSA)
# - double-check with CV_RMSE (more stable)
# Typically you'll pick the model with lowest test error as your "winner".


#---------------- 6) Train final model on full data for deployment --------------

# Example: if LASSO is best, refit on all rows of dat:

X_full <- model.matrix(int_rate ~ ., data = dat)[, -1]
y_full <- dat$int_rate

set.seed(1)
best_lasso_full <- cv.glmnet(X_full, y_full, alpha = 1)

# Save final model for Reality Check script
saveRDS(best_lasso_full, file = "best_model_lasso.rds")