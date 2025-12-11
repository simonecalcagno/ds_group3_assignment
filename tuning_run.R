library(tfruns)

runs <- tuning_run(
  "nn_experiment.R",
  runs_dir = "tuning_ffn_7layer",
  sample   = 0.7,   # you can reduce to 0.7 if needed
  flags = list(
    # Optimizer hyperparameters
    learning_rate = c(0.0005, 0.0002),
    batch_size    = c(256),
    
    # Architecture scaling factor
    width_factor  = c(0.75, 1.0),
    
    # Activation
    act = c("relu", "elu"),
    
    # Single dropout hyperparameter
    drop = c(0.1, 0.2, 0.3),
    
    epochs = 2000
  )
)



# ---- Find and print best run ----
library(tfruns)
library(dplyr)

# Falls du tuning_run() gerade ausgeführt hast, hast du bereits ein Objekt `runs`.
# Andernfalls könntest du die Runs z.B. so laden:
# runs <- ls_runs(runs_dir = "tuning_cnn")

best_10 <- runs %>%
  as.data.frame() %>%                      # sicherstellen, dass es ein Data Frame ist
  arrange(metric_val_loss) %>%             # nach val_loss sortieren (aufsteigend)
  select(
    starts_with("flag_"),                  # alle Hyperparameter (Setup)
    metric_val_accuracy,                   # KPI: Validation Accuracy
    metric_val_loss                        # KPI: Validation Loss
  ) %>%
  slice(1:10)                              # nur die 10 besten

print(best_10)


