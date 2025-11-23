# Function to preprocess loan application data for prediction
preprocess_lc <- function(raw) {
  # Make sure needed packages are loaded
  if (!requireNamespace("dplyr", quietly = TRUE)) stop("Please install dplyr")
  if (!requireNamespace("lubridate", quietly = TRUE)) stop("Please install lubridate")
  if (!requireNamespace("stringr", quietly = TRUE)) stop("Please install stringr")
  if (!requireNamespace("caret", quietly = TRUE)) stop("Please install caret")
  
  library(dplyr)
  library(lubridate)
  library(stringr)
  library(caret)
  
  dat <- raw
  
  # ---------------------- Drop attributes not available for new applications ----------------------
  drop_now <- c(
    "collection_recovery_fee","installment","initial_list_status", "funded_amnt","funded_amnt_inv",
    "issue_d","last_pymnt_amnt","last_pymnt_d","loan_status","next_pymnt_d",
    "out_prncp","out_prncp_inv","pymnt_plan","recoveries","total_pymnt",
    "total_pymnt_inv","total_rec_int","total_rec_late_fee","total_rec_prncp"
  )
  dat <- dat %>% select(-any_of(drop_now))
  
  # ---------------------- Drop IDs, URL and description ----------------------
  dat <- dat %>% select(-any_of(c("id", "member_id", "url", "desc")))
  
  # ---------------------- Text / high-cardinality categoricals ----------------------
  dat <- dat %>% select(-any_of(c("emp_title", "title", "zip_code")))
  
  # ---------------------- Binary flags for missing derog/delinquency/record ----------------------
  dat <- dat %>%
    mutate(
      has_major_derog   = ifelse(is.na(mths_since_last_major_derog), 0, 1),
      has_delinq        = ifelse(is.na(mths_since_last_delinq), 0, 1),
      has_public_record = ifelse(is.na(mths_since_last_record), 0, 1)
    ) %>%
    select(-mths_since_last_major_derog,
           -mths_since_last_delinq,
           -mths_since_last_record)
  
  # ---------------------- Drop derived/redundant variables ----------------------
  dat <- dat %>% select(-any_of(c("sub_grade", "policy_code")))
  
  # ---------------------- Joint applications ----------------------
  dat <- dat %>%
    mutate(is_joint = ifelse(application_type == "JOINT", 1, 0)) %>%
    select(-annual_inc_joint, -dti_joint, -verification_status_joint)
  dat <- dat %>% select(-is_joint)
  
  # ---------------------- Remove “block dependency” columns ----------------------
  block_vars <- c("open_acc_6m","open_il_6m","open_il_12m","open_il_24m",
                  "mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m",
                  "open_rv_24m","max_bal_bc","all_util","inq_fi","total_cu_tl","inq_last_12m")
  dat <- dat %>% select(-any_of(block_vars))
  
  # ---------------------- Remove block dependency rows (lack of credit history) ----------------------
  block_25 <- c("delinq_2yrs","inq_last_6mths","open_acc",
                "pub_rec","total_acc","acc_now_delinq")
  dat <- dat %>% filter(!if_any(all_of(block_25), is.na))
  
  # ---------------------- Block dependency rows II (credit summary) ----------------------
  block_63k <- c("tot_coll_amt","tot_cur_bal","total_rev_hi_lim")
  dat <- dat %>%
    mutate(has_credit_summary = ifelse(is.na(tot_coll_amt), 0, 1)) %>%
    mutate(across(all_of(block_63k),
                  ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
  
  # ---------------------- Replace NAs in revol_bal with median ----------------------
  dat <- dat %>%
    mutate(revol_bal = ifelse(is.na(revol_bal),
                              median(revol_bal, na.rm = TRUE),
                              revol_bal))
  
  # ---------------------- Replace NAs in revol_util with 0 ----------------------
  dat <- dat %>%
    mutate(revol_util = ifelse(is.na(revol_util), 0, revol_util))
  
  # ---------------------- Replace NAs in collections_12_mths_ex_med with 0 ----------------------
  dat <- dat %>%
    mutate(collections_12_mths_ex_med = ifelse(is.na(collections_12_mths_ex_med),
                                               0,
                                               collections_12_mths_ex_med))
  
  # ---------------------- Convert selected categoricals to factors ----------------------
  factor_cols <- c("term","grade","emp_length","home_ownership","verification_status",
                   "purpose","addr_state","initial_list_status","application_type")
  factor_cols <- intersect(factor_cols, names(dat))
  dat <- dat %>% mutate(across(all_of(factor_cols), as.factor))
  
  # ---------------------- Drop last_credit_pull_d ----------------------
  dat <- dat %>% select(-last_credit_pull_d)
  
  # ---------------------- Date handling: earliest_cr_line -> age in years ----------------------
  dat$earliest_cr_line <- lubridate::my(dat$earliest_cr_line)
  dat <- dat %>%
    mutate(
      earliest_cr_line_age_years =
        lubridate::time_length(lubridate::interval(earliest_cr_line, lubridate::today()), "years")
    ) %>%
    select(-earliest_cr_line)
  
  # ---------------------- Employment flag ----------------------
  dat <- dat %>%
    mutate(emp_length = na_if(stringr::str_to_lower(emp_length), "n/a")) %>%
    mutate(
      employed_flag = ifelse(is.na(emp_length), 0, 1)
    ) %>%
    select(-emp_length)
  
  # ---------------------- Home ownership grouping + dummies ----------------------
  dat <- dat %>%
    mutate(
      home_ownership = case_when(
        home_ownership %in% c("ANY", "NONE", "OTHER") ~ "OTHER",
        TRUE ~ as.character(home_ownership)
      )
    ) %>%
    mutate(home_ownership = as.factor(home_ownership))
  
  dummies_home <- caret::dummyVars(" ~ home_ownership", data = dat)
  home_dummies <- predict(dummies_home, newdata = dat)
  dat <- cbind(dat, home_dummies) %>%
    select(-home_ownership)
  
  # ---------------------- Purpose grouping + dummies ----------------------
  dat <- dat %>%
    mutate(
      purpose = dplyr::case_when(
        purpose %in% c("debt_consolidation", "credit_card",
                       "home_improvement", "other") ~ as.character(purpose),
        TRUE ~ "other"
      ),
      purpose = as.factor(purpose)
    )
  
  dummies_purpose <- caret::dummyVars(" ~ purpose", data = dat)
  purpose_dummies <- predict(dummies_purpose, newdata = dat)
  dat <- cbind(dat, purpose_dummies) %>%
    select(-purpose)
  
  
  # ----- Outliers and transformations -----
  
  # ---------------------- Annual income: cut extremes and log-transform ----------------------
  q_low  <- quantile(dat$annual_inc, 0.01, na.rm = TRUE)
  q_high <- quantile(dat$annual_inc, 0.99, na.rm = TRUE)
  
  dat <- dat %>%
    filter(annual_inc > 0 & annual_inc <= q_high) %>%
    mutate(annual_inc_log = log1p(annual_inc)) %>%
    select(-annual_inc)
  
  # ---------------------- Cut DTI, delinq_2yrs, cap inq_last_6mths, consistency open_acc / total_acc ----------------------
  dat <- dat %>% filter(dti <= 100)
  dat <- dat %>% filter(delinq_2yrs <= 10)
  dat <- dat %>% mutate(inq_last_6mths = pmin(inq_last_6mths, 6))
  dat <- dat %>% filter(open_acc <= total_acc)
  
  # ---------------------- pub_rec -> binary flag ----------------------
  dat <- dat %>%
    mutate(pub_rec_flag = ifelse(pub_rec > 0, 1, 0)) %>%
    select(-pub_rec)
  
  # ---------------------- Verification status dummies ----------------------
  dummies_verif <- caret::dummyVars(" ~ verification_status", data = dat)
  verification_dummies <- predict(dummies_verif, newdata = dat)
  dat <- cbind(dat, verification_dummies) %>%
    select(-verification_status)
  
  # ---------------------- Drop baseline dummies to avoid perfect multicollinearity ----------------------
  dat <- dat %>%
    select(
      -any_of(c("verification_status.Not Verified",
                "purpose.other",
                "home_ownership.RENT"))
    )
  
  # Return the fully preprocessed data frame (reality_flag kept if present)
  dat
}
