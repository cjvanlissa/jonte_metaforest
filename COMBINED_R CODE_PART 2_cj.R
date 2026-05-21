# ==========================================================
# Unified MetaForest Workflow: 3-Level Backbone + Bells & Whistles
# ==========================================================
#Set Working Directory

# ---------- Install/load packages ----------
# install.packages(c("readxl","dplyr","ranger","metafor","rfPermute","tibble",
#                     "ggplot2","pdp","vip","DALEX","fastshap"))
# library(readxl); library(dplyr); library(ranger); library(metafor)
# library(rfPermute); library(tibble); library(ggplot2)
# library(pdp); library(vip); library(DALEX); library(fastshap)
# install.packages("tibble")   # run this if not already installed
# library(tibble)
# install.packages("rfPermute")   # only once
# library(rfPermute)
#
# set.seed(123)
#
# # ---------- Load and prepare ----------
# dat <- read_excel("Metadat2.xlsx", sheet = "Sheet1", skip = 1) %>%
#   mutate(id_exp = factor(id_exp), ES_ID = factor(ES_ID))

predictors <- c("I_ILI", "I_MD", "I_CM", "I_SC", "I_NL",
                "I_SF", "I_CA", "I_TF", "I_UF", "I_GM", "I_WP", "I_GP", "I_IP",
                "I_CF", "I_EC", "I_IN", "I_SD", "CRIT", "ETH", "GEN", "IMP",
                "DES", "ASSN", "CTRL", "CTRY", "ERA", "MEA", "FOI", "FUND", "SET",
                "GRP", "FRQ", "SESS", "DUR", "GRD", "OPER", "OUT")
dat[predictors] <- lapply(dat[predictors], factor)

dat_rf <- dat %>%
  select(yi, vi, id_exp, ES_ID, all_of(predictors)) %>%
  na.omit()

######## ---------- Stage A: three-level meta model ----------######
m3 <- rma.mv(
  yi = yi,
  V  = vi,
  random = ~ 1 | id_exp/ES_ID,
  data = dat_rf,
  method = "REML"
)
dat_rf$resid_3L <- residuals(m3, type = "response")
wts <- sqrt(1 / dat_rf$vi)

# ---------- Zilong Code: export residuals & weights from the exact meta-analytic model ----------
library(metafor)
library(dplyr)

# Your exact 3-level meta-analytic model
m3 <- rma.mv(
  yi = yi,
  V  = vi,
  random = ~ 1 | id_exp/ES_ID,
  data = dat_rf,
  method = "REML"
)

dat_rf <- dat_rf %>%
  mutate(
    resid_3L = residuals(m3, type = "response"),
    wts = sqrt(1 / vi)   # same as in your ranger call
  )

# TOT HIER

# ---------- Stage B: forest on residuals ----------Model A vi as weight
rf_mod <- ranger(
  x = dat_rf %>% select(all_of(predictors)),
  y = dat_rf$resid_3L,
  case.weights = wts,
  importance = "permutation",
  num.trees = 1000,
  mtry = max(1, floor(sqrt(length(predictors)))),
  min.node.size = 5,
  respect.unordered.factors = "partition",
  seed = 123
)
vi_core <- sort(rf_mod$variable.importance, decreasing = TRUE)
vi_tbl  <- data.frame(variable = names(vi_core), perm_importance = vi_core)

# oob_r2 <- 1 - (rf_mod$prediction.error / var(dat_rf$resid_3L))
# oob_r2

# ---------- Zilong Code ----------Model B vi as predictor

rf_mod_B <- ranger(
  x = dat_rf %>% select(all_of(predictors), vi),  # X1–X37 + vi
  y = dat_rf$resid_3L,
  # no case.weights here
  importance = "permutation",
  num.trees = 1000,
  mtry = max(1, floor(sqrt(length(predictors) + 1))),  # +1 feature for vi
  min.node.size = 5,
  respect.unordered.factors = "partition",
  seed = 123
)
vi_tbl_Model_B <- enframe(
  sort(rf_mod_B$variable.importance, decreasing = TRUE),
  name = "variable", value = "perm_importance"
)


# ---------- rfPermute significance ----------
rfp <- rfPermute(
  x = dat_rf %>% select(all_of(predictors)),
  y = dat_rf$resid_3L,
  case.weights = wts,
  ntree = 500, nrep = 200, seed = 123
)
imp_mat <- importance(rfp)
imp_tbl <- tibble(
  variable = rownames(imp_mat),
  incMSE   = imp_mat[, "%IncMSE"],
  p_value  = imp_mat[, "%IncMSE.pval"]
)

write.csv(imp_mat, "Importance_pre_vi.csv", row.names = TRUE)



#### ---------- Model B: add vi_as_pred ----------####
dat_rf2 <- dat_rf %>%
  mutate(vi_as_pred = vi) %>%
  select(resid_3L, all_of(predictors), vi_as_pred)

rf_with_vi <- ranger(
  x = dat_rf2 %>% select(-resid_3L),
  y = dat_rf2$resid_3L,
  case.weights = wts,
  importance = "permutation",
  num.trees = 1000,
  seed = 123
)
vi_with_vi <- enframe(sort(rf_with_vi$variable.importance, decreasing = TRUE),
                      name = "variable", value = "perm_importance_with_vi")

write.csv(vi_with_vi, "vi_with_vi_results.csv", row.names = FALSE)



# ---------- Combined comparison table ----------
comparison_full <- vi_tbl %>%
  left_join(imp_tbl, by = "variable") %>%
  full_join(vi_with_vi, by = "variable") %>%
  mutate(delta = perm_importance_with_vi - perm_importance) %>%
  arrange(desc(coalesce(perm_importance_with_vi, perm_importance)))

print(comparison_full, n = Inf)
write.csv(comparison_full, "comparison_full.csv", row.names = FALSE)

# R-squared Calculations for Stage 2 Models#####
# ---------- For the main model (rf_mod) ----------
# Get predictions from the main Random Forest model
dat_rf$predicted_main <- predict(rf_mod, data = dat_rf)$predictions

# Calculate R-squared for the main model
ss_total_main <- sum((dat_rf$resid_3L - mean(dat_rf$resid_3L))^2)
ss_residual_main <- sum((dat_rf$resid_3L - dat_rf$predicted_main)^2)
r_squared_main <- 1 - (ss_residual_main / ss_total_main)

# Calculate weighted R-squared (accounting for sampling variance weights)
weighted_ss_total <- sum(wts * (dat_rf$resid_3L - weighted.mean(dat_rf$resid_3L, wts))^2)
weighted_ss_residual <- sum(wts * (dat_rf$resid_3L - dat_rf$predicted_main)^2)
weighted_r_squared_main <- 1 - (weighted_ss_residual / weighted_ss_total)

# ---------- For the sensitivity model (rf_with_vi) ----------
# Get predictions from the sensitivity model
dat_rf2$predicted_sensitivity <- predict(rf_with_vi, data = dat_rf2)$predictions

# Calculate R-squared for the sensitivity model
ss_total_sens <- sum((dat_rf2$resid_3L - mean(dat_rf2$resid_3L))^2)
ss_residual_sens <- sum((dat_rf2$resid_3L - dat_rf2$predicted_sensitivity)^2)
r_squared_sens <- 1 - (ss_residual_sens / ss_total_sens)

# Calculate weighted R-squared for sensitivity model
weighted_ss_total_sens <- sum(wts * (dat_rf2$resid_3L - weighted.mean(dat_rf2$resid_3L, wts))^2)
weighted_ss_residual_sens <- sum(wts * (dat_rf2$resid_3L - dat_rf2$predicted_sensitivity)^2)
weighted_r_squared_sens <- 1 - (weighted_ss_residual_sens / weighted_ss_total_sens)

# ---------- Create a summary table ----------
r2_summary <- data.frame(
  Model = c("Main Model (X1-X37)", "Sensitivity Model (X1-X37 + vi)"),
  R_squared = c(r_squared_main, r_squared_sens),
  Weighted_R_squared = c(weighted_r_squared_main, weighted_r_squared_sens),
  MSE = c(mean((dat_rf$resid_3L - dat_rf$predicted_main)^2),
          mean((dat_rf2$resid_3L - dat_rf2$predicted_sensitivity)^2)),
  RMSE = c(sqrt(mean((dat_rf$resid_3L - dat_rf$predicted_main)^2)),
           sqrt(mean((dat_rf2$resid_3L - dat_rf2$predicted_sensitivity)^2))),
  MAE = c(mean(abs(dat_rf$resid_3L - dat_rf$predicted_main)),
          mean(abs(dat_rf2$resid_3L - dat_rf2$predicted_sensitivity)))
)

# Print the results
cat("R-squared and Performance Metrics for Stage 2 Models:\n")
print(r2_summary)

# Write to CSV
write.csv(r2_summary, "stage2_model_performance_testing_ZP_0110.csv", row.names = FALSE)


# ---------- Additional: Out-of-bag (OOB) R-squared ----------
# For Random Forest, we can also use the OOB R-squared
oob_r2_main <- rf_mod$r.squared
oob_r2_sens <- rf_with_vi$r.squared

cat("\nOut-of-Bag R-squared values:\n")
cat("Main Model OOB R-squared:", oob_r2_main, "\n")
cat("Sensitivity Model OOB R-squared:", oob_r2_sens, "\n")

# ---------- Interpretation ----------
cat("\nInterpretation:\n")
cat("The R-squared values indicate how much of the residual heterogeneity (from the 3-level model)\n")
cat("is explained by the moderators. An R-squared of", round(r_squared_main, 3), "means that",
    round(r_squared_main * 100, 1), "% of the unexplained variance\n")
cat("in effect sizes is accounted for by the moderators X1-X37.\n")

if (r_squared_sens > r_squared_main) {
  cat("Adding sampling variance (vi) as a predictor increases R-squared by",
      round(r_squared_sens - r_squared_main, 3), "suggesting sampling variance\n")
  cat("explains additional heterogeneity in the effect sizes.\n")
} else {
  cat("Adding sampling variance (vi) as a predictor does not substantially improve model fit.\n")
}

# ---------- Visualization: Actual vs Predicted ----------
actual_vs_pred <- ggplot(dat_rf, aes(x = predicted_main, y = resid_3L)) +
  geom_point(alpha = 0.6, aes(size = 1/sqrt(vi))) +  # Size points by precision
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Actual vs Predicted Residuals (Main Model)",
       x = "Predicted Residuals",
       y = "Actual Residuals",
       subtitle = paste("R² =", round(r_squared_main, 3),
                        "| Weighted R² =", round(weighted_r_squared_main, 3))) +
  theme_minimal()

ggsave("actual_vs_predicted.png", actual_vs_pred, width = 8, height = 6)

# ---------- Residual distribution ----------
residual_plot <- ggplot(dat_rf, aes(x = resid_3L - predicted_main)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Distribution of Residual Errors",
       x = "Prediction Error (Actual - Predicted)",
       y = "Frequency",
       subtitle = paste("Mean Absolute Error:", round(mean(abs(dat_rf$resid_3L - dat_rf$predicted_main)), 3))) +
  theme_minimal()

ggsave("residual_distribution.png", residual_plot, width = 8, height = 6)

##### ---------- Bootstrap stability ----------################
B <- 500
vars <- predictors
rank_mat_base <- matrix(NA_real_, nrow = length(vars), ncol = B,
                        dimnames = list(vars, paste0("b",1:B)))
for (b in 1:B) {
  keep_ids <- sample(unique(dat_rf$id_exp), replace = TRUE)
  samp <- dat_rf %>% filter(id_exp %in% keep_ids)
  if (nrow(samp) < 20) next
  wts_b <- sqrt(1 / samp$vi)
  rf_b <- ranger(
    x = samp %>% select(all_of(vars)),
    y = samp$resid_3L,
    case.weights = wts_b,
    importance = "permutation",
    num.trees = 600
  )
  imp_b <- rf_b$variable.importance
  imp_b <- imp_b[match(vars, names(imp_b))]
  rank_mat_base[, b] <- rank(-imp_b, ties.method = "average")
}
stab_tbl_base <- tibble(
  variable = rownames(rank_mat_base),
  mean_rank = rowMeans(rank_mat_base, na.rm = TRUE),
  sd_rank   = apply(rank_mat_base, 1, sd, na.rm = TRUE),
  prop_top5 = rowMeans(apply(rank_mat_base, 2, function(r) r <= 5), na.rm = TRUE),
  model     = "Baseline (X1–X37)"
)

# With vi_as_pred
dat_rf2 <- dat_rf %>%
  mutate(vi_as_pred = vi) %>%
  select(id_exp, vi, resid_3L, all_of(predictors), vi_as_pred)
vars_vi <- c(predictors, "vi_as_pred")
rank_mat_vi <- matrix(NA_real_, nrow = length(vars_vi), ncol = B,
                      dimnames = list(vars_vi, paste0("b",1:B)))
for (b in 1:B) {
  keep_ids <- sample(unique(dat_rf2$id_exp), replace = TRUE)
  samp <- dat_rf2 %>% filter(id_exp %in% keep_ids)
  if (nrow(samp) < 20) next
  wts_b <- sqrt(1 / samp$vi)
  rf_b <- ranger(
    x = samp %>% select(all_of(vars_vi)),
    y = samp$resid_3L,
    case.weights = wts_b,
    importance = "permutation",
    num.trees = 600
  )
  imp_b <- rf_b$variable.importance
  imp_b <- imp_b[match(vars_vi, names(imp_b))]
  rank_mat_vi[, b] <- rank(-imp_b, ties.method = "average")
}
stab_tbl_vi <- tibble(
  variable = rownames(rank_mat_vi),
  mean_rank = rowMeans(rank_mat_vi, na.rm = TRUE),
  sd_rank   = apply(rank_mat_vi, 1, sd, na.rm = TRUE),
  prop_top5 = rowMeans(apply(rank_mat_vi, 2, function(r) r <= 5), na.rm = TRUE),
  model     = "With vi_as_pred"
)

stab_comparison <- bind_rows(stab_tbl_base, stab_tbl_vi)
print(stab_comparison, n = 40)
write.csv(stab_comparison, "bootstrap_stability.csv", row.names = FALSE)

# ==========================================================
# This: Section Optional: Bells & Whistles on 3-Level Forest - COMPLETE FIXED VERSION
# ==========================================================

# Ensure we only use predictors that were in the rf_mod (X1–X37)
valid_predictors <- predictors

# Pick the top predictor
top_predictor <- comparison_full %>%
  filter(variable %in% valid_predictors) %>%
  arrange(desc(perm_importance)) %>%
  slice(1) %>%
  pull(variable)

# 1. PDP for top predictor
# Use the original training data AS-IS (with factors)
pdp_data <- partial(
  rf_mod,
  pred.var = top_predictor,
  train = dat_rf %>% select(all_of(valid_predictors)),
  type = "regression"
)

# Convert to numeric ONLY for plotting, after calculating PDP
pdp_data[[top_predictor]] <- as.numeric(as.character(pdp_data[[top_predictor]]))

ggsave("pdp_top_predictor.png",
       ggplot(pdp_data, aes(x = .data[[top_predictor]], y = yhat)) +
         geom_line(linewidth = 1.5, color = "steelblue") +
         labs(title = paste("Partial Dependence:", top_predictor),
              x = top_predictor,
              y = "Predicted Residual (3L)") +
         theme_minimal(), width = 6, height = 4)

# 2. Interaction plot for top 2 predictors
top_two <- comparison_full %>%
  filter(variable %in% valid_predictors) %>%
  arrange(desc(perm_importance)) %>%
  slice(1:2) %>%
  pull(variable)

interaction_plot_data <- partial(
  rf_mod,
  pred.var = top_two,
  train = dat_rf %>% select(all_of(valid_predictors)),
  type = "regression"
)

# For the interaction plot, handle both variables appropriately
ggsave("interaction_top2.png",
       ggplot(interaction_plot_data, aes(x = .data[[top_two[1]]], y = yhat,
                                         color = .data[[top_two[2]]],
                                         group = .data[[top_two[2]]])) +
         geom_line(linewidth = 1.2) +
         labs(title = paste("Interaction:", top_two[1], "x", top_two[2]),
              y = "Predicted Residual (3L)") +
         theme_minimal(), width = 7, height = 5)

# 3. Cluster Analysis
set.seed(123)
effect_clusters <- kmeans(dat_rf$resid_3L, centers = 3)
dat_rf$cluster <- factor(effect_clusters$cluster)
cluster_profiles <- dat_rf %>%
  group_by(cluster) %>%
  summarize(
    n = n(),
    mean_resid = mean(resid_3L),
    sd_resid = sd(resid_3L),
    across(all_of(top_two), ~ mean(as.numeric(.)), .names = "mean_{.col}")
  )
write.csv(cluster_profiles, "cluster_profiles.csv", row.names = FALSE)

# 4. Complete DALEX Implementation
# Training features
X_train <- dat_rf %>%
  select(all_of(predictors)) %>%
  as.data.frame()
y_train <- dat_rf$resid_3L

# Define prediction wrapper for ranger
predict_function_ranger <- function(model, newdata) {
  predict(model, data = newdata)$predictions
}

# Build explainer for rf_mod - use DALEX::explain explicitly
explainer <- DALEX::explain(
  model = rf_mod,
  data  = X_train,
  y     = y_train,
  label = "3L MetaForest",
  predict_function = predict_function_ranger
)

# 4a. DALEX Variable Importance
dalex_importance <- model_parts(explainer)
importance_plot <- plot(dalex_importance) +
  ggtitle("DALEX Feature Importance: 3L MetaForest") +
  theme_minimal()

ggsave("dalex_importance.png", importance_plot, width = 10, height = 6)

# 4b. DALEX Partial Dependence Profiles
# Create PDP for top 3 variables using DALEX
top_three <- comparison_full %>%
  filter(variable %in% valid_predictors) %>%
  arrange(desc(perm_importance)) %>%
  slice(1:3) %>%
  pull(variable)

# Generate PDP for each top variable
pdp_plots <- list()
for (var in top_three) {
  pdp <- model_profile(explainer, variables = var)
  pdp_plots[[var]] <- plot(pdp) +
    ggtitle(paste("DALEX Partial Dependence:", var)) +
    theme_minimal()
  ggsave(paste0("dalex_pdp_", var, ".png"), pdp_plots[[var]], width = 6, height = 4)
}

# 4c. DALEX Break-Down Plots for specific observations
# Get a few representative observations from each cluster
representative_obs <- dat_rf %>%
  group_by(cluster) %>%
  slice(1:2) %>%
  ungroup() %>%
  select(all_of(predictors)) %>%
  as.data.frame()

# Create break-down explanations for each representative observation
bd_plots <- list()
for (i in 1:nrow(representative_obs)) {
  observation <- representative_obs[i, , drop = FALSE]
  breakdown <- predict_parts(explainer,
                             new_observation = observation,
                             type = "break_down")
  bd_plots[[i]] <- plot(breakdown) +
    ggtitle(paste("Break-Down Explanation for Observation", i,
                  "(Cluster:", dat_rf$cluster[i], ")")) +
    theme_minimal()
  ggsave(paste0("breakdown_obs_", i, ".png"), bd_plots[[i]], width = 10, height = 6)
}

# 4d. Model Performance Check
model_performance <- model_performance(explainer)
performance_plot <- plot(model_performance) +
  ggtitle("Model Performance Diagnostics") +
  theme_minimal()

ggsave("model_performance.png", performance_plot, width = 8, height = 6)

# Print performance metrics
cat("Model Performance Metrics:\n")
print(model_performance)

# 4e. SHAP Values using fastshap (separate from DALEX)
# Calculate SHAP values for a subset of data
shap_subset <- dat_rf %>%
  sample_n(min(100, nrow(dat_rf))) %>%  # Use smaller sample for efficiency
  select(all_of(predictors)) %>%
  as.data.frame()

# Define prediction function for fastshap
predict_function_fastshap <- function(object, newdata) {
  predict(object, data = newdata)$predictions
}

# Calculate SHAP values using fastshap
shap_values <- fastshap::explain(
  rf_mod,
  X = shap_subset,
  nsim = 50,  # Reduced for speed; increase for better accuracy
  pred_wrapper = predict_function_fastshap
)

# Plot SHAP summary
shap_summary_plot <- autoplot(shap_values) +
  ggtitle("SHAP Value Summary Plot") +
  theme_minimal()

ggsave("shap_summary.png", shap_summary_plot, width = 10, height = 8)

# 5. Additional Diagnostic: Residuals vs Predicted plot
dat_rf$predicted <- predict(rf_mod, data = dat_rf)$predictions

residual_plot <- ggplot(dat_rf, aes(x = predicted, y = resid_3L)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", color = "blue") +
  labs(title = "Residuals vs Predicted Values",
       x = "Predicted Residuals",
       y = "Actual Residuals (from 3L model)") +
  theme_minimal()

ggsave("residuals_vs_predicted.png", residual_plot, width = 8, height = 6)

# 6. Save final dataset with clusters and predictions
final_data <- dat_rf %>%
  select(yi, vi, id_exp, ES_ID, resid_3L, predicted, cluster, all_of(top_three))

write.csv(final_data, "final_analysis_data.csv", row.names = FALSE)

cat("\nBells & Whistles analysis complete! Check the generated PNG files and CSV outputs.\n")
