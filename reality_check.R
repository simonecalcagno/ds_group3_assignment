################################################################################
# run_model_on_secret_data.R
# Inference script for Assignment 2
################################################################################

set.seed(1)

################################################################################
# Libraries
################################################################################

library(dplyr)
library(keras3)
library(tensorflow)
library(fastDummies)

################################################################################
# 1. Load trained model
################################################################################

model <- load_model("final_model_ass2_group3.keras")

################################################################################
# 2. Load secret dataset
################################################################################

secret <- read.csv("Dataset-part-2-secret.csv", stringsAsFactors = FALSE)

# Remove technical ID if present
if ("ID" %in% names(secret)) {
  secret <- secret %>% select(-ID)
}

################################################################################
# 3. preprocessing
################################################################################

# Replace empty strings with NA
for (col in names(secret)) {
  if (is.character(secret[[col]])) {
    secret[[col]][secret[[col]] == ""] <- NA
  }
}

# Handle DAYS_EMPLOYED placeholder
secret$is_unemployed <- ifelse(secret$DAYS_EMPLOYED == 365243, 1L, 0L)
secret$DAYS_EMPLOYED[secret$DAYS_EMPLOYED == 365243] <- NA

# Create age and years employed
secret$AGE_YEARS <- abs(secret$DAYS_BIRTH) / 365
secret$YEARS_EMPLOYED <- abs(secret$DAYS_EMPLOYED) / 365

# Cap outliers
secret$AGE_YEARS <- pmin(secret$AGE_YEARS, 90)
secret$YEARS_EMPLOYED <- pmin(secret$YEARS_EMPLOYED, 60)

# Drop original day-based columns
secret <- secret %>% select(-DAYS_BIRTH, -DAYS_EMPLOYED)

# Treat missing occupation as category
secret$OCCUPATION_TYPE[is.na(secret$OCCUPATION_TYPE)] <- "Unknown"

# Cap counts
secret$CNT_FAM_MEMBERS <- pmin(secret$CNT_FAM_MEMBERS, 10)
secret$CNT_CHILDREN   <- pmin(secret$CNT_CHILDREN, 6)

# Remove uninformative constant feature
if ("FLAG_MOBIL" %in% names(secret)) {
  secret <- secret %>% select(-FLAG_MOBIL)
}

################################################################################
# 4. Dummy encoding
################################################################################

cat_cols <- names(secret)[sapply(secret, is.character) | sapply(secret, is.factor)]

secret_dummies <- fastDummies::dummy_cols(
  secret,
  select_columns = cat_cols,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

x_secret <- data.matrix(secret_dummies)

################################################################################
# 5. Handle missing values (robust fallback)
################################################################################

# Any remaining NA → 0 (safe fallback for inference)
x_secret[is.na(x_secret)] <- 0

################################################################################
# 6. Scaling (robust min–max fallback)
################################################################################


mins <- apply(x_secret, 2, min)
maxs <- apply(x_secret, 2, max)

same <- which(maxs == mins)
maxs[same] <- mins[same] + 1

x_secret_scaled <- sweep(x_secret, 2, mins, "-")
x_secret_scaled <- sweep(x_secret_scaled, 2, (maxs - mins), "/")

################################################################################
# 7. Prediction
################################################################################

pred_probs <- model %>% predict(x_secret_scaled)
pred_class <- apply(pred_probs, 1, which.max) - 1L

################################################################################
# 8. Map numeric predictions to original labels
################################################################################

status_levels <- c("0", "1", "2", "3", "4", "5", "C", "X")
pred_labels <- status_levels[pred_class + 1L]

################################################################################
# 9. Save predictions
################################################################################

output <- data.frame(
  prediction = pred_labels
)

write.csv(output, "predictions_secret_dataset.csv", row.names = FALSE)

cat("\nPredictions saved to predictions_secret_dataset.csv\n")
