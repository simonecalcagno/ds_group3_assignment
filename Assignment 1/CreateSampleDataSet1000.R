set.seed(123)   # for reproducibility

# 1. Load the full dataset
raw <- read_delim("LCdata.csv",
                  delim = ";",
                  na = c("", "NA","N/A"))

# 2. Randomly sample 1000 rows
idx_sample <- sample(nrow(raw), 1000)
sample_1000 <- raw[idx_sample, ]

# 3. Save to a new CSV file
write.csv(sample_1000, "LCdata_sample1000.csv", row.names = FALSE)