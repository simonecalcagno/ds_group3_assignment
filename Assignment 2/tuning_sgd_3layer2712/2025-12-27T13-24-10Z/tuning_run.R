############################################################
# Hyperparameter Tuning Script for Neural Network
############################################################
library(tfruns)



############################################################
# Run Hyperparameter Search
############################################################

runs <- tuning_run(
  "nn_experiment.R",
  runs_dir = "tuning_sgd_3layer2712",
  sample   = 1.0,  # Run all combinations (20 total)
  flags = list(
    # Optimizer hyperparameters
    learning_rate = c(0.05),      # 2 options
    batch_size    = c(128),    
    l2_reg        = c(0.00001),# 1 option (fixed)
    
    
    # Architecture scaling factor
    #width_factor  = c(1.5),          # 2 options
    
    # Activation function
    act = c("relu"),                 # 2 options
    
    
    # Training epochs
    epochs = 5000  # Let early stopping handle this
  )
)


############################################################
# Analyze Results - Sort by Validation Accuracy
############################################################

cat("\n========================================\n")
cat("TOP 10 RUNS BY VALIDATION ACCURACY\n")
cat("========================================\n\n")

best_by_acc <- runs %>%
  as.data.frame() %>%
  arrange(desc(metric_val_accuracy)) %>%  # Sort by accuracy descending
  select(
    run_dir,                         # Run identifier
    starts_with("flag_"),            # All hyperparameters
    metric_val_accuracy,             # Main metric: Validation Accuracy
    metric_val_loss,                 # Validation Loss
    metric_accuracy,                 # Training Accuracy (check overfitting)
    metric_loss                      # Training Loss
  ) %>%
  slice(1:10)

print(best_by_acc)

############################################################
# Alternative View: Sort by Validation Loss
############################################################

cat("\n========================================\n")
cat("TOP 10 RUNS BY VALIDATION LOSS\n")
cat("========================================\n\n")

best_by_loss <- runs %>%
  as.data.frame() %>%
  arrange(metric_val_loss) %>%      # Sort by loss ascending (lower is better)
  select(
    run_dir,
    starts_with("flag_"),
    metric_val_loss,
    metric_val_accuracy
  ) %>%
  slice(1:10)

print(best_by_loss)

############################################################
# Check for Overfitting
############################################################

cat("\n========================================\n")
cat("OVERFITTING CHECK (Top 5 by Accuracy)\n")
cat("========================================\n\n")

overfitting_check <- runs %>%
  as.data.frame() %>%
  arrange(desc(metric_val_accuracy)) %>%
  mutate(
    acc_gap = metric_accuracy - metric_val_accuracy  # Gap between train and val accuracy
  ) %>%
  select(
    run_dir,
    #flag_drop,
    #flag_width_factor,
    metric_val_accuracy,
    acc_gap
  ) %>%
  slice(1:5)

print(overfitting_check)
cat("\nNote: Large gaps indicate overfitting. Consider higher dropout or lower width_factor.\n")

############################################################
# Save Best Configuration
############################################################

best_run <- runs %>%
  as.data.frame() %>%
  arrange(desc(metric_val_accuracy)) %>%
  slice(1)

cat("\n========================================\n")
cat("BEST CONFIGURATION (by val_accuracy)\n")
cat("========================================\n\n")

cat("Run Directory:", best_run$run_dir, "\n")
cat("Validation Accuracy:", round(best_run$metric_val_accuracy, 4), "\n")
cat("Validation Loss:", round(best_run$metric_val_loss, 4), "\n\n")

cat("Best Hyperparameters:\n")
cat("  - learning_rate:", best_run$flag_learning_rate, "\n")
cat("  - batch_size:", best_run$flag_batch_size, "\n")
#cat("  - width_factor:", best_run$flag_width_factor, "\n")
cat("  - activation:", best_run$flag_act, "\n")
#cat("  - dropout:", best_run$flag_drop, "\n")

# Save best configuration to file
best_config <- list(
  learning_rate = best_run$flag_learning_rate,
  batch_size = best_run$flag_batch_size,
  #width_factor = best_run$flag_width_factor,
  act = best_run$flag_act,
  #drop = best_run$flag_drop,
  val_accuracy = best_run$metric_val_accuracy,
  val_loss = best_run$metric_val_loss,
  run_dir = best_run$run_dir
)

saveRDS(best_config, "best_hyperparameters.rds")
cat("\nBest configuration saved to: best_hyperparameters.rds\n")


############################################################
# Load and Evaluate Best Model on Test Set
############################################################

cat("\n========================================\n")
cat("NEXT STEPS\n")
cat("========================================\n\n")
cat("1. Review the best hyperparameters above\n")
cat("2. Uncomment the test evaluation section in nn_experiment.R\n")
cat("3. Run the best configuration on the full dataset:\n\n")
cat("   library(tfruns)\n")
cat("   training_run(\n")
cat("     'nn_experiment.R',\n")
cat("     flags = list(\n")
cat("       learning_rate =", best_run$flag_learning_rate, ",\n")
cat("       batch_size =", best_run$flag_batch_size, ",\n")
#cat("       width_factor =", best_run$flag_width_factor, ",\n")
cat("       act = '", best_run$flag_act, "',\n", sep="")
#cat("       drop =", best_run$flag_drop, ",\n")
cat("       epochs = 5000\n")
cat("     )\n")
cat("   )\n\n")
cat("4. This will give you the final test set performance\n")

