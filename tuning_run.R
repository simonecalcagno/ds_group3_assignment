library(tfruns)

runs <- tuning_run(
  "nn_experiment.R",
  runs_dir ="tuning_cnn_fine",
  sample   = 0.8,           # ca. 25% der Kombinationen
  flags = list(
    learning_rate = c(0.0005),  #0.0001, 0.00005,
    batch_size    = c(512),
    units1        = c(256, 512),
    units2        = c(64, 128),
    units3        = c(16, 32), 
    # units4        = c(16), # FIXIERT
    act1          = c("relu"),
    act2          = c("relu"),     # FIXIERT
    act3          = c("relu"),     # FIXIERT
    # act4          = c("relu"),   
    dropout       = c(0.1, 0.2),
    epochs        = c(2000)        # FIXIERT, Early Stopping stoppt früher
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


