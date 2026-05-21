# Install metaforest. This needs to be done only once.
install.packages("metaforest")
# Load the metaforest package
library(metaforest)
install.packages("dplyr")
library(dplyr)
#
# library(readxl)
# metaf<- read_excel("metadata.xlsx", sheet = "Sheet1", skip = 1)
#
# # Set a seed for the random number generator,
# so analyses can be replicated exactly.
set.seed(62)

#Define Predictors
predictors <- paste0("X", 1:37)
metaf[predictors] <- lapply(metaf[predictors], factor)



metaf_rf <- metaf %>%
  select(yi, vi, id_exp, ES_ID, all_of(predictors)) %>%
  na.omit()
# Force all predictors to factors
metaf_rf [predictors] <- lapply(metaf_rf [predictors], factor)

# Double-check
str(metaf_rf [predictors])

# Build the formula dynamically
form <- as.formula(paste("yi ~", paste(predictors, collapse = " + ")))

# Run model with many trees to check convergence
check_conv <- MetaForest(form,
                         data = metaf_rf,
                         study = "id_exp",
                         whichweights = "random",
                         num.trees = 20000)

# Plot convergence trajectory
plot(check_conv)



# Model with 2000 trees for replication
mf_rep <- MetaForest(form,
                     data = metaf_rf,
                     study = "id_exp",
                     whichweights = "random",
                     num.trees = 2000)

# Recursive preselection
preselected <- preselect(mf_rep,
                         replications = 100,
                         algorithm = "recursive")

   # Plot results
plot(preselected)



# Retain moderators with positive variable importance in more than
# 50% of replications
retain_mods <- preselect_vars(preselected, cutoff = 0.3)

# Load caret
library(caret)
# Set up 10-fold clustered CV
grouped_cv <- trainControl(method = "cv",
                           index = groupKFold(metaf_rf$id_exp, k = 10))

# Set up a tuning grid
tuning_grid <- expand.grid(whichweights = c("random", "fixed", "unif"),
                           mtry = 2:6,
                           min.node.size = 2:6)

# X should contain only retained moderators, clustering variable, and vi
X <- metaf_rf[, c("id_exp", "vi", retain_mods)]

# Convert any character moderators to factors
X[] <- lapply(X, function(col) {
  if (is.character(col)) factor(col) else col
})

str(X)

# Train the model
mf_cv <- train(
  y = metaf_rf$yi,
  x = X,
  method = ModelInfo_mf(),
  trControl = grouped_cv,
  tuneGrid = tuning_grid,
  num.trees = 10000
)

# Extract R^2_cv
r2_cv <- mf_cv$results$Rsquared[which.min(mf_cv$results$RMSE)]



#-----zilong code below----

# --- SENSITIVITY ANALYSIS --- #

# 1. Sensitivity to variable selection cutoff
different_cutoffs <- c(0.2, 0.3, 0.4, 0.5)
sensitivity_cutoffs <- list()

for(cutoff in different_cutoffs) {
  cat("Testing cutoff =", cutoff, "...\n")

  retain_mods_cutoff <- preselect_vars(preselected, cutoff = cutoff)
  cat("Number of selected variables:", length(retain_mods_cutoff), "\n")

  # Skip if no variables selected
  if (length(retain_mods_cutoff) == 0) {
    cat("No variables selected with cutoff", cutoff, "- skipping\n")
    sensitivity_cutoffs[[as.character(cutoff)]] <- NULL
    next
  }

  # Re-run model with this cutoff
  X_cutoff <- metaf_rf[, c("id_exp", "vi", retain_mods_cutoff)]
  X_cutoff[] <- lapply(X_cutoff, function(col) {
    if (is.character(col)) factor(col) else col
  })

  # DYNAMIC tuning grid based on number of predictors
  n_predictors <- length(retain_mods_cutoff)
  max_mtry <- min(6, n_predictors)  # Don't exceed available predictors

  tuning_grid_dynamic <- expand.grid(
    whichweights = c("random", "fixed", "unif"),
    mtry = 2:max_mtry,
    min.node.size = 2:6
  )

  cat("Using tuning grid with mtry up to:", max_mtry, "\n")

  mf_cutoff <- train(
    y = metaf_rf$yi,
    x = X_cutoff,
    method = ModelInfo_mf(),
    trControl = grouped_cv,
    tuneGrid = tuning_grid_dynamic,
    num.trees = 5000  # Smaller for speed
  )

  sensitivity_cutoffs[[as.character(cutoff)]] <- list(
    model = mf_cutoff,
    n_vars = n_predictors,
    selected_vars = retain_mods_cutoff
  )

  cat("Completed cutoff =", cutoff, "\n\n")
}

# 2. Sensitivity to weighting method (simpler approach)
weighting_methods <- c("random", "fixed", "unif")
sensitivity_weights <- list()

cat("Testing weighting methods...\n")
for(weight_method in weighting_methods) {
  cat("Running with", weight_method, "weights...\n")

  mf_weight <- MetaForest(form,
                          data = metaf_rf,
                          study = "id_exp",
                          whichweights = weight_method,
                          num.trees = 5000)
  sensitivity_weights[[weight_method]] <- mf_weight

  cat("Completed", weight_method, "weights\n")
}

# 3. Compare variable importance across sensitivity models
# Extract and compare variable importance from different analyses
if (length(sensitivity_weights) > 0) {
  varimp_comparison <- data.frame(
    variable = names(sensitivity_weights$random$forest$variable.importance)
  )

  for(weight_method in weighting_methods) {
    varimp_comparison[[weight_method]] <-
      sensitivity_weights[[weight_method]]$forest$variable.importance
  }

  print("Variable importance comparison across weighting methods:")
  print(varimp_comparison)
} else {
  cat("No weighting sensitivity results to compare\n")
}

# 4. Check consistency of top predictors
if (length(sensitivity_weights) > 0) {
  top_predictors <- lapply(sensitivity_weights, function(model) {
    vi <- model$forest$variable.importance
    names(vi)[order(vi, decreasing = TRUE)][1:5]  # Top 5 predictors
  })

  print("Top predictors across weighting methods:")
  print(top_predictors)
}

# 5. Compare results across cutoffs
if (length(sensitivity_cutoffs) > 0) {
  cat("\n=== RESULTS ACROSS DIFFERENT CUTOFFS ===\n")
  for(cutoff in names(sensitivity_cutoffs)) {
    cat("Cutoff:", cutoff, "\n")
    cat("Number of variables:", sensitivity_cutoffs[[cutoff]]$n_vars, "\n")
    cat("Selected variables:", paste(sensitivity_cutoffs[[cutoff]]$selected_vars, collapse = ", "), "\n")
    cat("Best R-squared:", max(sensitivity_cutoffs[[cutoff]]$model$results$Rsquared), "\n\n")
  }
}


#----zilong code above----
