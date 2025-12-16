library(tfruns)

runs <- tuning_run(
  "nn_experiment.R",
  runs_dir ="tuning_cnn_fine",
  sample   = 0.4,           # % of possible combinations to try
  flags = list(
    learning_rate = c(0.0001),  #0.0001, 0.00005,
    batch_size    = c(256),
    units1        = c(256),
    units2        = c(128, 256),
    units3        = c(64, 128), 
    units4        = c(32, 64), 
    #    units5        = c(256), 
    #    units6        = c(128), 
    
    act1          = c("relu"),
    act2          = c("relu"),   
    act3          = c("relu"),       
    act4          = c("relu"),   
    #    act5          = c("relu"),   
    #    act6          = c("relu"),   
    
    dropout1       = c(0.2),
    dropout2       = c(0, 0.1),
    dropout3       = c(0.2),
    dropout4       = c(0, 0.1),
    #    dropout5       = c(0.2),
    #    dropout6       = c(0, 0.2),
    
    epochs        = c(5000)        # FIXIERT, Early Stopping stoppt früher
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
