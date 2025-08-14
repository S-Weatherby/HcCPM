# Modeling Script - Script 03
# Author: Shelita Smith
# Date: August 2025
# Purpose: Predictive modeling for insurance charges using engineered features
# Goals: Multiple model comparison, validation, interpretation, and deployment preparation

# 0 Setup ####

## libraries
library(tidyverse)
library(caret)           # For model training and evaluation
library(randomForest)    # Random Forest (modeling)
library(glmnet)          # Ridge/Lasso regression
library(e1071)           # SVM (model)
library(gbm)             # Gradient Boosting
library(corrplot)        # Correlation visualization
library(VIM)             # Missing data visualization
library(pROC)            # ROC curves (if classification)
library(ModelMetrics)    # Additional metrics
library(pdp)             # Partial dependence plots

# reference tables
insurance_basic_features <- read_csv("data/processed/insurance_basic_features.csv")
complete_feature_analysis <- read_csv("outputs/tables/complete_feature_analysis_final.csv")
top_features <- read_csv("outputs/tables/top_performing_features.csv")

## High impact data
train_high <- read_csv("data/processed/train_high_impact.csv")
test_high <- read_csv("data/processed/test_high_impact.csv")
train_high_scaled <- read_csv("data/processed/train_high_impact_scaled.csv")  
test_high_scaled <- read_csv("data/processed/test_high_impact_scaled.csv")    

## Essential data - forgoing as features are already covered by high impact

# 1 Data Exploration (Training Data) ####
## 1.1 Data Summary ####
### 1.1.1 High Impact 
dataset_summary <- tibble(
  dataset = "High Impact Training Data",
  n_features = ncol(train_high) - 1,
  n_observations = nrow(train_high),
  target_mean = mean(train_high$charges),
  target_sd = sd(train_high$charges)
)

### 1.1.2 High Impact Scaled 
dataset_summary_hi_scaled <- tibble(
  dataset = "High Impact Scaled Training Data",
  n_features = ncol(train_high_scaled) - 1,
  n_observations = nrow(train_high_scaled),
  target_mean = mean(train_high_scaled$charges),
  target_sd = sd(train_high_scaled$charges)
)

## 1.2 Target Variable Analysis ####
### 1.2.1 High Impact
charges_analysis <- train_high %>%
  summarise(
    mean_charges = mean(charges),
    median_charges = median(charges),
    sd_charges = sd(charges),
    skewness = moments::skewness(charges),
    kurtosis = moments::kurtosis(charges),
    min_charges = min(charges),
    max_charges = max(charges)
  )

### 1.2.2 High Impact Scaled TVA
charges_analysis_hi_scaled <- train_high_scaled %>%
  summarise(
    mean_charges = mean(charges),
    median_charges = median(charges),
    sd_charges = sd(charges),
    min_charges = min(charges),
    max_charges = max(charges)
  )

## 1.3 Feature Correlation/ Data Leakage Features ####
# Define explicit data leakage features (derived from charges)
leakage_features <- c(
  "charges_percentile_rank",  # Percentile rank derived from charges
  "smoker_cost_multiplier",   # Cost multiplier derived from charges
  "age_cost_curve"           # Cost curve derived from charges
)

### 1.3.1 High Impact 
numeric_features <- train_high %>% select_if(is.numeric)
cor_matrix <- cor(numeric_features, use = "complete.obs")
high_cor_features_hi <- findCorrelation(cor_matrix, cutoff = 0.8, names = TRUE)
existing_leakage_hi <- leakage_features[leakage_features %in% names(train_high)]

# Remove leakage features
train_high <- train_high %>%
  select(-all_of(existing_leakage_hi))

### 1.3.2 High Impact Scaled Feature Corr
numeric_features_hi_scaled <- train_high_scaled %>% select_if(is.numeric)
cor_matrix_hi_scaled <- cor(numeric_features_hi_scaled, use = "complete.obs")
high_cor_features_hi_scaled <- findCorrelation(cor_matrix_hi_scaled, cutoff = 0.8, names = TRUE)
existing_leakage_hi_scaled <- leakage_features[leakage_features %in% names(train_high_scaled)]

# Remove leakage features
train_high_scaled <- train_high_scaled %>%
  select(-all_of(existing_leakage_hi_scaled))

# Apply same cleaning to test sets
test_high <- test_high %>%
  select(-all_of(existing_leakage_hi))

test_high_scaled <- test_high_scaled %>%
  select(-all_of(existing_leakage_hi_scaled))

# 2 Data Pre-processing ####
set.seed(123)  # For reproducibility 

## 2.1 Validation Split ####
# Validation/Test Split for UNSCALED data
val_idx <- createDataPartition(test_high$charges, p = 0.50, list = FALSE)
validation_data <- test_high[val_idx, ]
final_test_data <- test_high[-val_idx, ]

# Validation/Test Split for SCALED data 
val_idx_scaled <- createDataPartition(test_high_scaled$charges, p = 0.50, list = FALSE)
validation_data_scaled <- test_high_scaled[val_idx_scaled, ]      
final_test_data_scaled <- test_high_scaled[-val_idx_scaled, ]    

## 2.2 Training Missing Values Check(MVC) ####

### 2.1.1 High Impact Missing Values check
missing_summary <- train_high %>%
  summarise_all(~sum(is.na(.))) %>%
  gather(variable, missing_count) %>%
  mutate(missing_percentage = missing_count / nrow(train_high) * 100) %>%
  filter(missing_count > 0)

missing_summary_scaled <- train_high_scaled %>%
  summarise_all(~sum(is.na(.))) %>%
  gather(variable, missing_count) %>%
  mutate(missing_percentage = missing_count / nrow(train_high_scaled) * 100) %>%
  filter(missing_count > 0)

# 3 Model Training and Validation ####
## Linear Regression (baseline performance, scaled)
## Regularized Regression (Ridge, Lasso, and Elastic Net; good for over fitting, scaled)
## Random Forest (feature importance validation, scale-invariant)
## XGBoost (gradient boosting benchmark, scale invariant)
## 3.1 Linear Models ####
# Features w/ eta-squared (effect size) > 0.6
# top 5 features= Simple
# full features

### features w/ 5 highest effect sizes = Simple
top_features_clean <- top_features %>%
  filter(!str_detect(feature_name, "→|×")) %>%  # Remove ANOVA descriptions with arrows
  filter(feature_name %in% colnames(train_high)) %>%  # Only existing columns
  arrange(desc(eta_squared))

top_5_features <- top_features_clean$feature_name[1:5] 
formula_simple <- as.formula(paste("charges ~", paste(top_5_features, collapse = " + ")))
### compound_lifestyle_risk_score not in essential data; as effect-size is more impactful, foregoing essential data as a modeling set 
# available_features <- top_5_features[top_5_features %in% names(train_high_ess_scaled)]
# formula_essential <- as.formula(paste("charges ~", paste(available_features, collapse = " + ")))

### 3.1.1 HI Scaled Simple LM top 5 features
lm_hi_scaled_simp <- lm(formula_simple, data = train_high_scaled)
model_summary_simple <- summary(lm_hi_scaled_simp)

# Predictions and evaluation
hi_scaled_lm_pred_val <- predict(lm_hi_scaled_simp, validation_data_scaled)
lm_rmse <- RMSE(hi_scaled_lm_pred_val, validation_data_scaled$charges)
lm_r2 <- R2(hi_scaled_lm_pred_val, validation_data_scaled$charges)

# save simple diagnostic plots
png("outputs/plots/simple_linear_diagnostics.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(lm_hi_scaled_simp, main = "Simple Linear Model Diagnostics") ##generate 4 diagnostic plots: Residuals vs fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage
dev.off()

### 3.1.2 Full LM Model
lm_full <- lm(charges ~ ., data = train_high_scaled)
model_summary_full <- summary(lm_full)

# Predictions and evaluation on validation data
hi_scaled_lm_full_pred_val <- predict(lm_full, validation_data_scaled)
lm_full_rmse <- RMSE(hi_scaled_lm_full_pred_val, validation_data_scaled$charges)
lm_full_r2 <- R2(hi_scaled_lm_full_pred_val, validation_data_scaled$charges)

# Save full model diagnostic plots
png("outputs/plots/full_linear_diagnostics.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(lm_full, main = "Full Linear Model Diagnostics")
dev.off()

### 3.1.3
# Stepwise selection
lm_step <- step(lm_full, direction = "both")
model_summary_step <- summary(lm_step)

# Save stepwise model diagnostic plots
png("outputs/plots/stepwise_linear_diagnostics.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
plot(lm_step, main = "Stepwise Linear Model Diagnostics")
dev.off()

# Evaluate stepwise model
step_pred_val <- predict(lm_step, validation_data_scaled)
step_rmse <- RMSE(step_pred_val, validation_data_scaled$charges)
step_r2 <- R2(step_pred_val, validation_data_scaled$charges)

### 3.1.4 Linear Model Comparison
# Linear models comparison
#### diagnostic plots from linear training must be ran for the remaining code to run
linear_model_results <- data.frame(
  Model = c("Simple (Top 5)", "Stepwise", "Full Model"),
  Features = c(length(top_5_features), 
               length(coef(lm_step)) - 1,
               length(coef(lm_full)) - 1),
  RMSE = c(lm_rmse, step_rmse, 
           RMSE(predict(lm_full, validation_data_scaled), validation_data_scaled$charges)),
  R_squared = c(lm_r2, step_r2,
                R2(predict(lm_full, validation_data_scaled), validation_data_scaled$charges)),
  Adj_R_squared = c(model_summary_simple$adj.r.squared, 
                    model_summary_step$adj.r.squared,
                    model_summary_full$adj.r.squared),
  AIC = c(AIC(lm_hi_scaled_simp), AIC(lm_step), AIC(lm_full))
)

# Round for readability
linear_model_results$RMSE <- round(linear_model_results$RMSE, 2)
linear_model_results$R_squared <- round(linear_model_results$R_squared, 4)
linear_model_results$Adj_R_squared <- round(linear_model_results$Adj_R_squared, 4)
linear_model_results$AIC <- round(linear_model_results$AIC, 1)

# Save linear model comparison
write.csv(linear_model_results, "outputs/tables/linear_model_comparison.csv", row.names = FALSE)

# Linear models performance comparison plot
png("outputs/plots/linear_models_performance.png", width = 1500, height = 500, res = 100)
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))

plot(validation_data_scaled$charges, hi_scaled_lm_pred_val,
     main = paste("Simple Model (R² =", round(lm_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("blue", 0.6))
abline(0, 1, col = "red", lwd = 2)

plot(validation_data_scaled$charges, step_pred_val,
     main = paste("Stepwise Model (R² =", round(step_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("green", 0.6))
abline(0, 1, col = "red", lwd = 2)

full_pred_val <- predict(lm_full, validation_data_scaled)
full_r2 <- R2(full_pred_val, validation_data_scaled$charges)
plot(validation_data_scaled$charges, full_pred_val,
     main = paste("Full Model (R² =", round(full_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("orange", 0.6))
abline(0, 1, col = "red", lwd = 2)

dev.off()

## 3.2 Regularized Models ####
### 3.2.1 Regularized Setup

# Prepare data for glmnet (using scaled data)
x_train <- model.matrix(charges ~ ., train_high_scaled)[,-1]
y_train <- train_high_scaled$charges
x_val <- model.matrix(charges ~ ., validation_data_scaled)[,-1]
y_val <- validation_data_scaled$charges

### 3.2.2 Ridge (adds penalty for large coefficients, good for correlated predictors)
# Ridge Regression with cross-validation
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)

# Ridge coefficient plot 
ridge_plot <- plot(ridge_model) #5 star plot generated

ridge_pred <- predict(ridge_model, x_val, s = "lambda.min")
ridge_rmse <- RMSE(ridge_pred, y_val)
ridge_r2 <- R2(ridge_pred, y_val)

### 3.2.3 Lasso

# Lasso Regression with cross-validation
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)

# Lasso coefficient plot
lasso_plot <- plot(lasso_model)

lasso_pred <- predict(lasso_model, x_val, s = "lambda.min")
lasso_rmse <- RMSE(lasso_pred, y_val)
lasso_r2 <- R2(lasso_pred, y_val)

### 3.2.4 Elastic
elastic_model <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 10)

# Elastic Net coefficient plot 
elastic_plot <- plot(elastic_model)

elastic_pred <- predict(elastic_model, x_val, s = "lambda.min")
elastic_rmse <- RMSE(elastic_pred, y_val)
elastic_r2 <- R2(elastic_pred, y_val)

### 3.2.5 Regularized Comparison
regularized_results <- data.frame(
  Model = c("Ridge", "Lasso", "Elastic Net"),
  RMSE = c(ridge_rmse, lasso_rmse, elastic_rmse),
  R_squared = c(ridge_r2, lasso_r2, elastic_r2),
  Lambda_Min = c(ridge_model$lambda.min, lasso_model$lambda.min, elastic_model$lambda.min),
  Features_Selected = c(
    sum(coef(ridge_model, s = "lambda.min")[-1] != 0),
    sum(coef(lasso_model, s = "lambda.min")[-1] != 0),
    sum(coef(elastic_model, s = "lambda.min")[-1] != 0)
  )
)

# Round for readability
regularized_results[,2:4] <- round(regularized_results[,2:4], 4)

write.csv(regularized_results, "outputs/tables/regularized_model_comparison.csv", row.names = FALSE)

# Regularized models performance plot
png("outputs/plots/regularized_models_performance.png", width = 1500, height = 500, res = 100)
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))

plot(y_val, ridge_pred, main = paste("Ridge (R² =", round(ridge_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("purple", 0.6))
abline(0, 1, col = "red", lwd = 2)

plot(y_val, lasso_pred, main = paste("Lasso (R² =", round(lasso_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("darkgreen", 0.6))
abline(0, 1, col = "red", lwd = 2)

plot(y_val, elastic_pred, main = paste("Elastic Net (R² =", round(elastic_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("brown", 0.6))
abline(0, 1, col = "red", lwd = 2)

dev.off()

## 3.3 Tree Models ####
### 3.3.1 Random Forests

# Tune Random Forest
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

rf_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE
)

rf_model <- train(
  charges ~ .,
  data = train_high,
  method = "rf",
  tuneGrid = rf_grid,
  trControl = rf_control,
  ntree = 500
)

# Feature importance plot
RF_plot <- plot(varImp(rf_model), top = 15) #appears to align well with the feature eta-squared (yay!)

# Predictions
rf_pred <- predict(rf_model, validation_data)
rf_rmse <- RMSE(rf_pred, validation_data$charges)
rf_r2 <- R2(rf_pred, validation_data$charges)

### 3.3.2 XGBoost 

# GBM tuning parameters
gbm_grid <- expand.grid(
  n.trees = c(100, 300, 500),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1, 0.2),
  n.minobsinnode = c(10, 20)
)

gbm_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE
)

gbm_model <- train(
  charges ~ .,
  data = train_high,
  method = "gbm",
  tuneGrid = gbm_grid,
  trControl = gbm_control,
  verbose = FALSE
)

# GBM tuning plot 
gbm_plot <- plot(gbm_model)

gbm_pred <- predict(gbm_model, validation_data)
gbm_rmse <- RMSE(gbm_pred, validation_data$charges)
gbm_r2 <- R2(gbm_pred, validation_data$charges)

# GBM feature importance
png("outputs/plots/gbm_feature_importance.png", width = 1500, height = 600, res = 100)
plot(varImp(gbm_model), top = 15, main = "GBM Feature Importance")
dev.off()

# GBM performance plot
png("outputs/plots/gbm_performance.png", width = 1200, height = 600, res = 100)
par(mfrow = c(1, 2))

plot(validation_data$charges, gbm_pred,
     main = paste("GBM (R² =", round(gbm_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("purple", 0.6))
abline(0, 1, col = "red", lwd = 2)

plot(gbm_pred, validation_data$charges - gbm_pred,
     main = "GBM Residuals",
     xlab = "Predicted", ylab = "Residuals", pch = 16, col = alpha("purple", 0.6))
abline(h = 0, col = "red", lwd = 2)

dev.off()

# Training history plot
png("outputs/plots/gbm_training_history.png", width = 1000, height = 600, res = 100)
plot(gbm_model$finalModel, main = "GBM Training Error by Iteration")
dev.off()

### 3.3.3 Tree Model Comparison

tree_results <- data.frame(
  Model = c("Random Forest", "GBM"),
  RMSE = c(rf_rmse, gbm_rmse),
  R_squared = c(rf_r2, gbm_r2),
  Best_Parameters = c(
    paste("mtry =", rf_model$bestTune$mtry),
    paste("trees =", gbm_model$bestTune$n.trees, ", depth =", gbm_model$bestTune$interaction.depth)
  )
)

# Round for readability
tree_results[,2:3] <- round(tree_results[,2:3], 4)

write.csv(tree_results, "outputs/tables/tree_model_comparison.csv", row.names = FALSE)

# Tree models performance plot
png("outputs/plots/tree_models_performance.png", width = 1000, height = 500, res = 100)
par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))

plot(validation_data$charges, rf_pred,
     main = paste("Random Forest (R² =", round(rf_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("forestgreen", 0.6))
abline(0, 1, col = "red", lwd = 2)

plot(validation_data$charges, gbm_pred,
     main = paste("GBM (R² =", round(gbm_r2, 3), ")"),
     xlab = "Actual", ylab = "Predicted", pch = 16, col = alpha("darkblue", 0.6))
abline(0, 1, col = "red", lwd = 2)

dev.off()

# 4 Model Comparison ####
## 4.1 Performance Comparison ####
all_models_comparison <- bind_rows(
  linear_model_results %>% select(Model, RMSE, R_squared) %>% mutate(Type = "Linear"),
  regularized_results %>% select(Model, RMSE, R_squared) %>% mutate(Type = "Regularized"),
  tree_results %>% select(Model, RMSE, R_squared) %>% mutate(Type = "Tree-based")
) %>% 
  arrange(RMSE) %>%
  mutate(
    Rank_RMSE = row_number(),
    MAE = case_when(
      Model == "Simple (Top 5)" ~ MAE(hi_scaled_lm_pred_val, validation_data_scaled$charges),
      Model == "Stepwise" ~ MAE(step_pred_val, validation_data_scaled$charges),
      Model == "Full Model" ~ MAE(full_pred_val, validation_data_scaled$charges),
      Model == "Ridge" ~ MAE(ridge_pred, y_val),
      Model == "Lasso" ~ MAE(lasso_pred, y_val),
      Model == "Elastic Net" ~ MAE(elastic_pred, y_val),
      Model == "Random Forest" ~ MAE(rf_pred, validation_data$charges),
      Model == "GBM" ~ MAE(gbm_pred, validation_data$charges)
    )
  )

# Round MAE
all_models_comparison$MAE <- round(all_models_comparison$MAE, 2)

write.csv(all_models_comparison, "outputs/tables/comprehensive_model_comparison.csv", row.names = FALSE)

# Comprehensive performance visualization
png("outputs/plots/comprehensive_model_comparison.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 2), mar = c(5, 4, 2, 1))

# RMSE comparison
barplot(all_models_comparison$RMSE, names.arg = all_models_comparison$Model,
        las = 2, main = "RMSE Comparison", ylab = "RMSE", col = rainbow(nrow(all_models_comparison)))

# R-squared comparison  
barplot(all_models_comparison$R_squared, names.arg = all_models_comparison$Model,
        las = 2, main = "R-squared Comparison", ylab = "R-squared", col = rainbow(nrow(all_models_comparison)))

# MAE comparison
barplot(all_models_comparison$MAE, names.arg = all_models_comparison$Model,
        las = 2, main = "MAE Comparison", ylab = "MAE", col = rainbow(nrow(all_models_comparison)))

# Performance by model type
boxplot(RMSE ~ Type, data = all_models_comparison, main = "RMSE by Model Type", ylab = "RMSE")

dev.off()

## 4.2 Cross-validation Comparison ####
# 10-fold CV for top performing models (caret models only)
cv_models <- list()

# Add models that were trained with caret (tree models)
if(exists("rf_model")) cv_models$RandomForest <- rf_model
if(exists("gbm_model")) cv_models$GBM <- gbm_model

if(length(cv_models) > 0) {
  cv_results <- resamples(cv_models)
  
  # CV results plot 
  bwplot(cv_results)
  
  # Save CV summary
  cv_summary <- summary(cv_results)
  write.csv(cv_summary$statistics$RMSE, "outputs/tables/cross_validation_results.csv")
}

## 4.3 RMSE Best Model ####
### RMSE - root mean square error <- avg distance between predicted and observed values 

##Best Model: GBM
best_model_name <- all_models_comparison$Model[1]
best_model_rmse <- all_models_comparison$RMSE[1]
best_model_r2 <- all_models_comparison$R_squared[1]

# Store the actual best model object
best_model <- switch(best_model_name,
                     "Simple (Top 5)" = lm_hi_scaled_simp,
                     "Stepwise" = lm_step,
                     "Full Model" = lm_full,
                     "Ridge" = ridge_model,
                     "Lasso" = lasso_model,
                     "Elastic Net" = elastic_model,
                     "Random Forest" = rf_model,
                     "GBM" = gbm_model
)

# Best model summary
best_model_summary <- data.frame(
  Metric = c("Model", "RMSE", "R-squared", "MAE", "Model Type"),
  Value = c(best_model_name, best_model_rmse, best_model_r2,
            all_models_comparison$MAE[1], all_models_comparison$Type[1])
)

write.csv(best_model_summary, "outputs/tables/best_model_summary.csv", row.names = FALSE)

# 5 Model Evaluation on Test Sets ####
## 5.1 Test Set Performance ####

# Get predictions from best model on test set ## Best Model = GBM
if(best_model_name %in% c("Ridge", "Lasso", "Elastic Net")) {
  x_test <- model.matrix(charges ~ ., final_test_data_scaled)[,-1]
  final_predictions <- predict(best_model, x_test, s = "lambda.min")
  final_predictions <- as.vector(final_predictions)
} else if(best_model_name %in% c("Random Forest", "GBM")) {
  final_predictions <- predict(best_model, final_test_data)
} else {
  final_predictions <- predict(best_model, final_test_data_scaled)
}

# Calculate final metrics
final_rmse <- RMSE(final_predictions, final_test_data$charges)
final_r2 <- R2(final_predictions, final_test_data$charges)
final_mae <- MAE(final_predictions, final_test_data$charges)

# Final performance summary
final_performance <- data.frame(
  Dataset = c("Validation", "Test"),
  RMSE = c(best_model_rmse, final_rmse),
  R_squared = c(best_model_r2, final_r2),
  MAE = c(all_models_comparison$MAE[1], final_mae)
)

write.csv(final_performance, "outputs/tables/final_model_performance.csv", row.names = FALSE)

## 5.2 Residual Analysis - linear models only/scaled data ####
# Comprehensive residual analysis
residuals <- final_test_data_scaled$charges - final_predictions

residual_analysis <- data.frame(
  actual = final_test_data_scaled$charges,
  predicted = final_predictions,
  residuals = residuals,
  residuals_standardized = residuals / sd(residuals),
  abs_residuals = abs(residuals),
  percent_error = abs(residuals) / final_test_data_scaled$charges * 100
)

# Residual analysis plots
png("outputs/plots/residual_analysis.png", width = 1500, height = 1000, res = 100)
par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

# Actual vs Predicted
plot(residual_analysis$actual, residual_analysis$predicted,
     main = "Actual vs Predicted", xlab = "Actual", ylab = "Predicted",
     pch = 16, col = alpha("blue", 0.6))
abline(0, 1, col = "red", lwd = 2)

# Residuals vs Predicted
plot(residual_analysis$predicted, residual_analysis$residuals,
     main = "Residuals vs Predicted", xlab = "Predicted", ylab = "Residuals",
     pch = 16, col = alpha("red", 0.6))
abline(h = 0, col = "black", lwd = 2)

# Q-Q Plot
qqnorm(residual_analysis$residuals_standardized, main = "Q-Q Plot of Standardized Residuals")
qqline(residual_analysis$residuals_standardized, col = "red", lwd = 2)

# Histogram of residuals
hist(residual_analysis$residuals, breaks = 30, main = "Distribution of Residuals",
     xlab = "Residuals", col = "lightblue", border = "black")

# Absolute residuals vs predicted
plot(residual_analysis$predicted, residual_analysis$abs_residuals,
     main = "Absolute Residuals vs Predicted", xlab = "Predicted", ylab = "Absolute Residuals",
     pch = 16, col = alpha("green", 0.6))

# Percent error distribution
hist(residual_analysis$percent_error, breaks = 30, main = "Percentage Error Distribution",
     xlab = "Percentage Error", col = "orange", border = "black")

dev.off()

# Residual statistics
residual_stats <- data.frame(
  Statistic = c("Mean Residual", "SD Residual", "Mean Absolute Error", "Mean Percent Error", 
                "Median Percent Error", "95th Percentile Error"),
  Value = c(mean(residual_analysis$residuals), sd(residual_analysis$residuals),
            mean(residual_analysis$abs_residuals), mean(residual_analysis$percent_error),
            median(residual_analysis$percent_error), quantile(residual_analysis$percent_error, 0.95))
)

residual_stats$Value <- round(residual_stats$Value, 3)
write.csv(residual_stats, "outputs/tables/residual_analysis_stats.csv", row.names = FALSE)

# 6 Model Interpretation ####

## 6. 1 Feature Importance Analysis ####
# Extract feature importance based on model type
if(best_model_name %in% c("Random Forest", "GBM")) {
  feature_importance <- varImp(best_model, scale = TRUE)
  importance_df <- data.frame(
    feature = rownames(feature_importance$importance),
    importance = feature_importance$importance[,1]
  ) %>%
    arrange(desc(importance))
} else if(best_model_name %in% c("Ridge", "Lasso", "Elastic Net")) {
  coefs <- coef(best_model, s = "lambda.min")
  importance_df <- data.frame(
    feature = rownames(coefs)[-1],  # Remove intercept
    importance = abs(coefs[-1,1])   # Absolute coefficients
  ) %>%
    arrange(desc(importance)) %>%
    filter(importance > 0)
} else {
  # For linear models, use absolute t-statistics
  model_summary <- summary(best_model)
  importance_df <- data.frame(
    feature = names(model_summary$coefficients[-1,1]),
    importance = abs(model_summary$coefficients[-1,3])  # t-statistics
  ) %>%
    arrange(desc(importance))
}

write.csv(importance_df, "outputs/tables/final_model_feature_importance.csv", row.names = FALSE)

# Feature importance plot
png("outputs/plots/feature_importance.png", width = 1000, height = 800, res = 100)
par(mar = c(5, 8, 2, 1))

top_features_plot <- head(importance_df, 15)
barplot(top_features_plot$importance, names.arg = top_features_plot$feature,
        horiz = TRUE, las = 1, main = paste("Top 15 Features -", best_model_name),
        xlab = "Importance", col = "steelblue")

dev.off()

## 6.2 Partial Dependence Plots ####
  ## Tree models only
  ### show the dependence between the target response and a set of input features of interest, marginalizing over the values of all other input features (the ‘complement’ features).
if(best_model_name %in% c("Random Forest", "GBM")) {
  top_5_important <- head(importance_df$feature, 5)
  
  # Filter to only features that exist in the training data
  available_features <- top_5_important[top_5_important %in% names(train_high)]
  
  # Create PDP plots for available features only
  for(i in 1:length(available_features)) {
    feature_name <- available_features[i]
    
    png(paste0("outputs/plots/pdp_", gsub("[^A-Za-z0-9]", "_", feature_name), ".png"), 
        width = 800, height = 600, res = 100)
    
    pdp_data <- pdp::partial(best_model, pred.var = feature_name, 
                             train = train_high, 
                             pred.fun = function(object, newdata) predict(object, newdata))
    
    plot(pdp_data, main = paste("Partial Dependence:", feature_name))
    
    dev.off()
  }
}


## 6.3 Business Impact Analysis ####
# Cost prediction accuracy by segments
  ##smoker, BMI, age, region, sex, age groups, has children, and risk profile

# Function to create segment analysis
analyze_segment <- function(data, segment_var, filename_suffix, plot_title_prefix) {
  segment_analysis <- data %>%
    mutate(
      predicted_charges = final_predictions,
      prediction_error = charges - predicted_charges,
      abs_prediction_error = abs(prediction_error),
      percentage_error = abs_prediction_error / charges * 100
    ) %>%
    group_by(across(all_of(segment_var))) %>%
    summarise(
      n_observations = n(),
      mean_actual = mean(charges),
      mean_predicted = mean(predicted_charges),
      mean_percentage_error = mean(percentage_error),
      median_percentage_error = median(percentage_error),
      rmse_segment = sqrt(mean(prediction_error^2)),
      .groups = "drop"
    )
  
  # Round for readability
  segment_analysis[,3:7] <- round(segment_analysis[,3:7], 2)
  
  # Save table
  write.csv(segment_analysis, paste0("outputs/tables/segment_analysis_", filename_suffix, ".csv"), row.names = FALSE)
  
  # Create plots
  png(paste0("outputs/plots/segment_analysis_", filename_suffix, ".png"), width = 1200, height = 800, res = 100)
  par(mfrow = c(2, 2), mar = c(5, 4, 3, 2))
  
  # Plot 1: Mean Percentage Error
  barplot(segment_analysis$mean_percentage_error, 
          names.arg = segment_analysis[[segment_var]],
          main = paste("Mean % Error by", plot_title_prefix), 
          ylab = "Mean % Error",
          col = rainbow(nrow(segment_analysis)),
          las = 2)
  
  # Plot 2: RMSE
  barplot(segment_analysis$rmse_segment, 
          names.arg = segment_analysis[[segment_var]],
          main = paste("RMSE by", plot_title_prefix), 
          ylab = "RMSE",
          col = heat.colors(nrow(segment_analysis)),
          las = 2)
  
  # Plot 3: Actual vs Predicted Means
  barplot(rbind(segment_analysis$mean_actual, segment_analysis$mean_predicted), 
          names.arg = segment_analysis[[segment_var]],
          main = paste("Actual vs Predicted by", plot_title_prefix),
          ylab = "Mean Charges",
          col = c("lightblue", "lightcoral"),
          legend = c("Actual", "Predicted"),
          beside = TRUE,
          las = 2)
  
  # Plot 4: Sample Sizes
  barplot(segment_analysis$n_observations, 
          names.arg = segment_analysis[[segment_var]],
          main = paste("Sample Size by", plot_title_prefix), 
          ylab = "Number of Observations",
          col = "lightgreen",
          las = 2)
  
  dev.off()
  
  return(segment_analysis)
}

# Analyze smoker segment
if("smoker" %in% names(final_test_data)) {
  smoker_analysis <- analyze_segment(final_test_data, "smoker", "smoker", "Smoking Status")
}

# Analyze region segment
if("region" %in% names(final_test_data)) {
  region_analysis <- analyze_segment(final_test_data, "region", "region", "Region")
}

# Analyze sex segment
if("sex" %in% names(final_test_data)) {
  sex_analysis <- analyze_segment(final_test_data, "sex", "sex", "Gender")
}

# Create and analyze age groups
if("age" %in% names(final_test_data)) {
  final_test_data <- final_test_data %>%
    mutate(age_group = cut(age, breaks = c(0, 30, 45, 65, Inf), 
                           labels = c("Young", "Middle", "Pre-Senior", "Senior")))
  age_analysis <- analyze_segment(final_test_data, "age_group", "age", "Age Group")
}

# Create and analyze BMI categories
if("bmi" %in% names(final_test_data)) {
  final_test_data <- final_test_data %>%
    mutate(bmi_category = cut(bmi, breaks = c(0, 18.5, 25, 30, Inf),
                              labels = c("Underweight", "Normal", "Overweight", "Obese")))
  bmi_analysis <- analyze_segment(final_test_data, "bmi_category", "bmi", "BMI Category")
}

# Create and analyze children categories
if("children" %in% names(final_test_data)) {
 final_test_data <- final_test_data %>%
    mutate(children_group = case_when(
      children == 0 ~ "No Children",
      children == 1 ~ "One Child",
      children >= 2 ~ "Multiple Children"
    ))
  children_analysis <- analyze_segment(test_data, "children_group", "children", "Children Status")
}

# Create and analyze cost tiers
if("charges" %in% names(final_test_data)) {
 final_test_data <-final_test_data %>%
    mutate(cost_tier = cut(charges, 
                           breaks = quantile(charges, c(0, 0.33, 0.67, 1), na.rm = TRUE),
                           labels = c("Low Cost", "Medium Cost", "High Cost"),
                           include.lowest = TRUE))
  cost_analysis <- analyze_segment(final_test_data, "cost_tier", "cost_tier", "Cost Tier")
}

# Create and analyze risk profiles (combination of smoker and BMI)
if(all(c("smoker", "bmi") %in% names(final_test_data))) {
 final_test_data <-final_test_data %>%
    mutate(risk_profile = case_when(
      smoker == "yes" & bmi >= 30 ~ "High Risk",
      smoker == "yes" | bmi >= 30 ~ "Medium Risk", 
      TRUE ~ "Low Risk"
    ))
  risk_analysis <- analyze_segment(final_test_data, "risk_profile", "risk", "Risk Profile")
}

# Summary comparison table across all segments
segment_summaries <- data.frame(
  Segment = character(),
  Category = character(),
  Mean_Percentage_Error = numeric(),
  RMSE = numeric(),
  Sample_Size = numeric(),
  stringsAsFactors = FALSE
)

# Collect results from each analysis
analysis_list <- list(
  if(exists("smoker_analysis")) list(name = "Smoker", data = smoker_analysis, var = "smoker"),
  if(exists("region_analysis")) list(name = "Region", data = region_analysis, var = "region"),
  if(exists("sex_analysis")) list(name = "Sex", data = sex_analysis, var = "sex"),
  if(exists("age_analysis")) list(name = "Age", data = age_analysis, var = "age_group"),
  if(exists("bmi_analysis")) list(name = "BMI", data = bmi_analysis, var = "bmi_category"),
  if(exists("children_analysis")) list(name = "Children", data = children_analysis, var = "children_group"),
  if(exists("cost_analysis")) list(name = "Cost Tier", data = cost_analysis, var = "cost_tier"),
  if(exists("risk_analysis")) list(name = "Risk Profile", data = risk_analysis, var = "risk_profile")
)

# Remove NULL entries
analysis_list <- analysis_list[!sapply(analysis_list, is.null)]

# Summary table
for(analysis in analysis_list) {
  temp_data <- analysis$data %>%
    mutate(Segment = analysis$name) %>%
    select(Segment, Category = all_of(analysis$var), 
           Mean_Percentage_Error = mean_percentage_error,
           RMSE = rmse_segment,
           Sample_Size = n_observations)
  
  segment_summaries <- rbind(segment_summaries, temp_data)
}

# Save summary table
if(nrow(segment_summaries) > 0) {
  write.csv(segment_summaries, "outputs/tables/segment_analysis_summary.csv", row.names = FALSE)
  
  # Create summary visualization
  png("outputs/plots/segment_analysis_summary.png", width = 1400, height = 800, res = 100)
  par(mfrow = c(1, 2), mar = c(8, 4, 3, 2))
  
  # Summary plot 1: Mean Percentage Error by segment
  avg_errors <- aggregate(Mean_Percentage_Error ~ Segment, segment_summaries, mean)
  barplot(avg_errors$Mean_Percentage_Error, 
          names.arg = avg_errors$Segment,
          main = "Average Percentage Error by Segment Type", 
          ylab = "Mean % Error",
          col = "skyblue",
          las = 2)
  
  # Summary plot 2: RMSE by segment  
  avg_rmse <- aggregate(RMSE ~ Segment, segment_summaries, mean)
  barplot(avg_rmse$RMSE, 
          names.arg = avg_rmse$Segment,
          main = "Average RMSE by Segment Type", 
          ylab = "RMSE",
          col = "lightgreen",
          las = 2)
  
  dev.off()
}

# 7 Model Deployment Prep ####

## 7.1 Model serialization ####
### RDS files

# Save final model
saveRDS(best_model, "models/final_insurance_cost_model.rds")

# Save preprocessing parameters
preprocessing_params <- list(
  model_name = best_model_name,
  feature_names = if(best_model_name %in% c("Random Forest", "GBM")) {
    names(train_high)[-1]
  } else {
    names(train_high_scaled)[-1]
  },
  scaling_needed = !best_model_name %in% c("Random Forest", "GBM"),
  train_statistics = if(best_model_name %in% c("Random Forest", "GBM")) {
    list(
      mean_charges = mean(train_high$charges),
      sd_charges = sd(train_high$charges)
    )
  } else {
    list(
      mean_charges = mean(train_high_scaled$charges),
      sd_charges = sd(train_high_scaled$charges),
      feature_means = sapply(train_high_scaled[,-1], mean),
      feature_sds = sapply(train_high_scaled[,-1], sd)
    )
  },
  top_features = head(importance_df$feature, 10)
)

saveRDS(preprocessing_params, "models/preprocessing_parameters.rds")

## 7.2 Prediction Function ####
# Create reusable prediction function
predict_insurance_cost <- function(new_data, 
                                   model_path = "models/final_insurance_cost_model.rds",
                                   params_path = "models/preprocessing_parameters.rds") {
  
  # Load model and parameters
  model <- readRDS(model_path)
  params <- readRDS(params_path)
  
  # Apply pre-processing (scaling if needed)
  processed_data <- new_data
  
  if(params$scaling_needed && !is.null(params$train_statistics$feature_means)) {
    for(col in names(params$train_statistics$feature_means)) {
      if(col %in% names(processed_data)) {
        processed_data[[col]] <- (processed_data[[col]] - params$train_statistics$feature_means[[col]]) / 
          params$train_statistics$feature_sds[[col]]
      }
    }
  }
  
  # Make predictions based on model type
  if(params$model_name %in% c("Ridge", "Lasso", "Elastic Net")) {
    x_new <- model.matrix(~ ., processed_data[,params$feature_names])[,-1]
    predictions <- predict(model, x_new, s = "lambda.min")
    predictions <- as.vector(predictions)
  } else {
    predictions <- predict(model, processed_data)
  }
  
  return(predictions)
}

saveRDS(predict_insurance_cost, "models/prediction_function.rds")

## 7.3 Model Documentation ####
model_documentation <- list(
  model_info = list(
    model_type = best_model_name,
    training_date = Sys.Date(),
    r_version = R.version.string,
    packages_used = c("caret", "randomForest", "glmnet", "e1071", "gbm", "tidyverse")
  ),
  performance_metrics = list(
    validation_rmse = best_model_rmse,
    validation_r2 = best_model_r2,
    test_rmse = final_rmse,
    test_r2 = final_r2,
    test_mae = final_mae
  ),
  feature_importance = importance_df,
  data_requirements = list(
    required_features = preprocessing_params$feature_names,
    scaling_required = preprocessing_params$scaling_needed,
    missing_data_handling = "Complete cases only"
  ),
  model_assumptions = list(
    "Assumes linear relationships for linear models",
    "Requires scaled numerical features for regularized models",
    "Performance validated on similar insurance data",
    "Model may not generalize to different populations"
  ),
  usage_guidelines = list(
    "Use for insurance cost prediction only",
    "Regularly monitor model performance",
    "Retrain quarterly or when performance degrades",
    "Validate predictions against business rules"
  ),
  limitations = list(
    "Based on historical data patterns",
    "May not capture future market changes",
    "Assumes data distribution remains stable",
    "Outlier predictions should be manually reviewed"
  )
)

saveRDS(model_documentation, "outputs/model_documentation.rds")

# 8 Executive Summary & Reporting ####
## 8.1 Exec Summary ####

executive_summary <- data.frame(
  Metric = c("Best Model", "Final Test RMSE", "Final Test R-squared", "Final Test MAE",
             "Top Contributing Factor", "Model Complexity", "Business Impact",
             "Deployment Ready"),
  Value = c(best_model_name, 
            round(final_rmse, 2),
            round(final_r2, 4),
            round(final_mae, 2),
            importance_df$feature[1],
            ifelse(best_model_name %in% c("Random Forest", "GBM"), "High", 
                   ifelse(best_model_name %in% c("Ridge", "Lasso", "Elastic Net"), "Medium", "Low")),
            paste("±", round(final_mae, 0), "average prediction error"),
            "Yes"
  )
)

write.csv(executive_summary, "outputs/tables/executive_summary.csv", row.names = FALSE)

## 8.2 Comprehensive Visualization ####

# Final dashboard-style plot
png("outputs/plots/final_model_dashboard.png", width = 1600, height = 1200, res = 100)
layout(matrix(c(1,1,2,3,4,4,5,6), nrow = 2, byrow = TRUE))

# Main performance plot
plot(residual_analysis$actual, residual_analysis$predicted,
     main = paste("Final Model Performance:", best_model_name),
     xlab = "Actual Charges", ylab = "Predicted Charges",
     pch = 16, col = alpha("blue", 0.6), cex = 0.8)
abline(0, 1, col = "red", lwd = 2)
text(min(residual_analysis$actual), max(residual_analysis$predicted),
     paste("R² =", round(final_r2, 3), "\nRMSE =", round(final_rmse, 0)),
     adj = c(0, 1), cex = 1.2, font = 2)

# Feature importance (top 10)
top_10_features <- head(importance_df, 10)
barplot(top_10_features$importance, names.arg = top_10_features$feature,
        main = "Top 10 Feature Importance", las = 2, cex.names = 0.8,
        col = "steelblue", border = "white")

# Residual distribution
hist(residual_analysis$residuals, breaks = 30, main = "Residual Distribution",
     xlab = "Residuals", col = "lightgreen", border = "white")
abline(v = 0, col = "red", lwd = 2)

# Model comparison (top 5)
top_5_models <- head(all_models_comparison, 5)
barplot(top_5_models$RMSE, names.arg = top_5_models$Model,
        main = "Top 5 Models by RMSE", las = 2, cex.names = 0.8,
        col = rainbow(5), border = "white")

# Error by prediction range
prediction_ranges <- cut(residual_analysis$predicted, breaks = 5, labels = c("Low", "Low-Med", "Medium", "Med-High", "High"))
error_by_range <- tapply(residual_analysis$percent_error, prediction_ranges, mean)
barplot(error_by_range, main = "Error by Prediction Range",
        ylab = "Mean % Error", col = "orange", border = "white")

# Performance timeline (RMSE progression)
model_rmse_timeline <- all_models_comparison$RMSE
plot(1:length(model_rmse_timeline), model_rmse_timeline, type = "b",
     main = "Model Performance Progression", xlab = "Model Number", ylab = "RMSE",
     pch = 16, col = "purple", lwd = 2)
points(1, model_rmse_timeline[1], col = "red", pch = 16, cex = 2)  # Best model

dev.off()

# 9 Model Monitoring and Future Work ####

## 9.1 Performance Monitoring Framework ####
# Monitoring benchmarks
monitoring_benchmarks <- data.frame(
  Metric = c("RMSE", "R_squared", "MAE", "Mean_Percent_Error"),
  Baseline_Value = c(final_rmse, final_r2, final_mae, mean(residual_analysis$percent_error)),
  Warning_Threshold = c(final_rmse * 1.1, final_r2 * 0.9, final_mae * 1.1, mean(residual_analysis$percent_error) * 1.2),
  Alert_Threshold = c(final_rmse * 1.2, final_r2 * 0.8, final_mae * 1.2, mean(residual_analysis$percent_error) * 1.5)
)

monitoring_benchmarks[,2:4] <- round(monitoring_benchmarks[,2:4], 4)

write.csv(monitoring_benchmarks, "outputs/tables/monitoring_benchmarks.csv", row.names = FALSE)

## 9.2 Enhancement Roadmap ####

enhancement_roadmap <- data.frame(
  Priority = c("High", "High", "Medium", "Medium", "Low"),
  Enhancement = c("Automated retraining pipeline", "Real-time prediction API", 
                  "Ensemble model implementation", "External data integration",
                  "Deep learning exploration"),
  Timeline = c("1 month", "2 months", "3 months", "6 months", "12 months"),
  Expected_Impact = c("Maintain accuracy", "Business efficiency", 
                      "Improved accuracy", "Better predictions", "Innovation")
)

write.csv(enhancement_roadmap, "outputs/tables/enhancement_roadmap.csv", row.names = FALSE)

# 10 Model Validation and Testing ####
## 10.1 Robustness Testing ####
# Test model on different data subsets
if(nrow(final_test_data) > 100) {
  # Random subsamples
  set.seed(456)
  subsample_indices <- sample(nrow(final_test_data), size = floor(nrow(final_test_data) * 0.5))
  subsample_data <- final_test_data[subsample_indices, ]
  
  # Get predictions for subsample
  if(best_model_name %in% c("Ridge", "Lasso", "Elastic Net")) {
    x_subsample <- model.matrix(charges ~ ., final_test_data_scaled[subsample_indices, ])[,-1]
    subsample_predictions <- predict(best_model, x_subsample, s = "lambda.min")
    subsample_predictions <- as.vector(subsample_predictions)
  } else if(best_model_name %in% c("Random Forest", "GBM")) {
    subsample_predictions <- predict(best_model, subsample_data)
  } else {
    subsample_predictions <- predict(best_model, final_test_data_scaled[subsample_indices, ])
  }
  
  # Calculate subsample performance
  subsample_rmse <- RMSE(subsample_predictions, subsample_data$charges)
  subsample_r2 <- R2(subsample_predictions, subsample_data$charges)
  
  robustness_test <- data.frame(
    Test = c("Full Test Set", "Random Subsample (50%)"),
    RMSE = c(final_rmse, subsample_rmse),
    R_squared = c(final_r2, subsample_r2),
    N_observations = c(nrow(final_test_data), nrow(subsample_data))
  )
  
  robustness_test[,2:3] <- round(robustness_test[,2:3], 4)
  
  write.csv(robustness_test, "outputs/tables/robustness_test_results.csv", row.names = FALSE)
}

## 10.2 Cross- Model Validation Summary ####

# Final model selection justification
model_selection_justification <- data.frame(
  Criterion = c("Lowest RMSE", "Highest R-squared", "Best Cross-Validation", 
                "Simplicity", "Interpretability", "Deployment Readiness"),
  Best_Model = c(
    all_models_comparison$Model[which.min(all_models_comparison$RMSE)],
    all_models_comparison$Model[which.max(all_models_comparison$R_squared)],
    ifelse(length(cv_models) > 0, "Available for caret models only", "N/A"),
    ifelse(any(grepl("Simple", all_models_comparison$Model)), "Simple (Top 5)", "Linear models"),
    ifelse(any(grepl("Linear|Simple", all_models_comparison$Model)), "Linear models", "Tree models have built-in importance"),
    "All models ready"
  ),
  Selected_Model = rep(best_model_name, 6),
  Rationale = c(
    "Primary performance metric",
    "Explains most variance",
    "Robust across folds",
    "Fewer parameters, easier to maintain",
    "Clear coefficient interpretation",
    "Serialized and documented"
  )
)

write.csv(model_selection_justification, "outputs/tables/model_selection_justification.csv", row.names = FALSE)

# 11 Script 04 Setup ####
## Savin  workspace for Script 4
# Create workspace directory 
if (!dir.exists("workspace")) {
  dir.create("workspace")
}

# Save all key objects for Script 4
save(
  # Data splits
  final_test_data,
  validation_data,
  train_high,
  final_test_data_scaled,
  validation_data_scaled,
  train_high_scaled,
  
  # Models
  best_model,
  best_model_name,
  lm_hi_scaled_simp,
  lm_step,
  lm_full,
  ridge_model,
  lasso_model,
  elastic_model,
  rf_model,
  gbm_model,
  
  # Predictions and results
  final_predictions,
  final_rmse,
  final_r2,
  final_mae,
  residual_analysis,
  
  # Feature importance
  importance_df,
  
  # Model comparison results
  all_models_comparison,
  linear_model_results,
  regularized_results,
  tree_results,
  
  # Cross-validation results (if they exist)
  cv_models,
  cv_results,
  
  # Other key objects
  preprocessing_params,
  
  file = "workspace/modeling_results.RData"
)

#12 Sample Data to Share

if (!dir.exists("data/sample_data")) {
  dir.create("data/sample_data")
}
# Sample data
sample_data <- train_high_scaled %>%
  select(-charges) %>%
  sample_n(50) %>%
  arrange(age_group_ordinal)
 
# Save for sharing
write.csv(sample_data, "data/sample_data/model_input_sample.csv", row.names = FALSE)

# Deatures used
feature_summary <- data.frame(
  feature_name = names(train_high_scaled)[-which(names(train_high_scaled) == "charges")],
  feature_type = sapply(train_high_scaled[,-which(names(train_high_scaled) == "charges")], class),
  example_value = sapply(train_high_scaled[,-which(names(train_high_scaled) == "charges")], function(x) x[1])
)

write.csv(feature_summary, "sample_data/feature_summary.csv", row.names = FALSE)