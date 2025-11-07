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

# cap everything above 100% dti
dat <- dat |> filter(dti <= 100)
# cap delinq_2yrs at 10
dat <- dat |> filter(delinq_2yrs <= 10)
# cap inq_last_6mths at 6
dat <- dat |> mutate(inq_last_6mths = pmin(inq_last_6mths, 6))
#remove errors where open_account > total_acc
dat <- dat |> filter(open_acc <= total_acc)

#comment
