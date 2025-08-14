#Feature Engineering Script - Script 02 (Debugged)
# Author: Shelita Smith
# Date: July 23, 2025 | August 01, 2025
# Purpose: Feature engineering with ANOVA analysis and comprehensive documentation
# Goals: ANOVA'd features, Analysis table guides, updated data dictionary in preparation for modeling

# 0 Set-Up ####

# Libraries
library(tidyverse)
library(fastDummies)
library(caret)
library(corrplot)

# Data dictionary setup
data_dictionary <- tibble(
  variable_name = character(),
  data_type = character(),
  source = character(),
  description = character(),
  possible_values = character(),
  missing_count = numeric(),
  missing_percent = numeric(),
  notes = character()
)

add_to_dictionary <- function(var_name, data_type, source, description, possible_values = "", notes = "") {
  new_row <- tibble(
    variable_name = var_name,
    data_type = data_type,
    source = source,
    description = description,
    possible_values = possible_values,
    missing_count = 0,
    missing_percent = 0,
    notes = notes
  )
  data_dictionary <<- bind_rows(data_dictionary, new_row)
}

update_dictionary <- function(var_name, notes) {
  if (var_name %in% data_dictionary$variable_name) {
    data_dictionary$notes[data_dictionary$variable_name == var_name] <<- notes
  }
}

# Ensure output directories exist
if (!dir.exists("outputs/tables")) dir.create("outputs/tables", recursive = TRUE)
if (!dir.exists("data/processed")) dir.create("data/processed", recursive = TRUE)

# Load data and resource tables (data dictionary, anova analysis tables)
insurance_clean <- read.csv("data/processed/insurance_clean.csv")
hcup_age_final <- read.csv("data/processed/hcup_age_clean.csv")

glimpse(insurance_clean)
glimpse(hcup_age_final)

# 1 Benchmark Summary ####

# HCUP data has multiple years per age group
# need ONE average per age group for easy merging
hcup_age_summary <- hcup_age_final %>%
  group_by(age_group_standard) %>%
  summarise(
    avg_hcup_charges = mean(avg_hospital_charges, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(hcup_age_summary, "outputs/tables/hcup_benchmark_summary.csv")

# Display benchmarks
print(hcup_age_summary)

# 2 Merge Insurance w/ Benchmarks ####

insurance_with_benchmarks <- insurance_clean %>%
  left_join(hcup_age_summary, by = "age_group_standard")

write.csv(insurance_with_benchmarks, "outputs/tables/insurance_with_benchmarks.csv")

# Check if the merge worked
glimpse(insurance_with_benchmarks)

# How many people got benchmarks?
merge_success_count <- sum(!is.na(insurance_with_benchmarks$avg_hcup_charges))
cat("Records successfully merged with benchmarks:", merge_success_count, "\n")

# 3 Train/Test Split ####
## prevents data leakage by ensuring feature engineering only uses training data

set.seed(123)  # For reproducibility

# stratified split based on charges quartiles
insurance_with_benchmarks$charges_quartile <- ntile(insurance_with_benchmarks$charges, 4)

# 70% train, 30% test (will split test further into validation/test in Script 3)
train_idx <- createDataPartition(insurance_with_benchmarks$charges_quartile, p = 0.70, list = FALSE)

train_data <- insurance_with_benchmarks[train_idx, ] %>% select(-charges_quartile)
test_data <- insurance_with_benchmarks[-train_idx, ] %>% select(-charges_quartile)

# Verify split distribution
split_verification <- data.frame(
  Dataset = c("Training", "Test", "Total"),
  N_Observations = c(nrow(train_data), nrow(test_data), nrow(insurance_with_benchmarks)),
  Mean_Charges = c(mean(train_data$charges), mean(test_data$charges), mean(insurance_with_benchmarks$charges)),
  Smoker_Pct = c(
    mean(train_data$smoker == "yes") * 100,
    mean(test_data$smoker == "yes") * 100,
    mean(insurance_with_benchmarks$smoker == "yes") * 100
  )
)

write_csv(split_verification, "outputs/tables/train_test_split_verification.csv")

# 4 Feature Engineering on Training Data Only ####

create_basic_features <- function(data) {
  # Only create features that don't use target information
  data_engineered <- data %>%
    mutate(
      # Simple encodings - NO target information
      smoker_encoded = as.numeric(smoker == "yes"),
      sex_encoded = as.numeric(sex == "male"),
      
      # Age groups (domain knowledge based)
      age_group_standard = case_when(
        age < 25 ~ "young_adult",
        age < 35 ~ "adult", 
        age < 45 ~ "middle_aged",
        age < 55 ~ "pre_senior",
        TRUE ~ "senior"
      ),
      
      # BMI categories (medical standards)
      bmi_category = case_when(
        bmi < 18.5 ~ "underweight",
        bmi < 25 ~ "normal",
        bmi < 30 ~ "overweight",
        TRUE ~ "obese"
      ),
      
      # Non-linear transformations
      age_squared = age^2,
      bmi_squared = bmi^2,
      age_cubed = age^3,
      age_log = log(pmax(age, 1)),
      bmi_log = log(pmax(bmi, 1)),
      
      # Family structure
      has_children = as.numeric(children > 0),
      children_squared = children^2,
      
      # Simple interactions (domain knowledge based)
      smoker_age_interaction = smoker_encoded * age,
      smoker_bmi_interaction = smoker_encoded * bmi,
      age_bmi_interaction = age * bmi,
      smoker_children_interaction = smoker_encoded * has_children,
      
      # Regional encoding (no cost information)
      region_southeast = as.numeric(region == "southeast"),
      region_southwest = as.numeric(region == "southwest"),
      region_northwest = as.numeric(region == "northwest"),
      # northeast as reference category
      
      # Age bins for analysis
      age_bins = cut(age, breaks = seq(15, 70, by = 5), include.lowest = TRUE),
      
      # BMI health categories
      bmi_health_category = case_when(
        bmi < 18.5 ~ "underweight",
        bmi < 25 ~ "normal",
        bmi < 30 ~ "overweight",
        TRUE ~ "obese"
      ),
      
      # Lifestyle risk indicators (no cost data)
      high_bmi_smoker = as.numeric(smoker_encoded == 1 & bmi >= 30),
      senior_smoker = as.numeric(smoker_encoded == 1 & age >= 55),
      
      # Ordinal encodings
      age_group_ordinal = case_when(
        age < 25 ~ 1,
        age < 35 ~ 2,
        age < 45 ~ 3,
        age < 55 ~ 4,
        TRUE ~ 5
      ),
      
      bmi_category_ordinal = case_when(
        bmi < 18.5 ~ 1,
        bmi < 25 ~ 2,
        bmi < 30 ~ 3,
        TRUE ~ 4
      )
    )
  
  return(data_engineered)
}

# Apply basic feature engineering to training data
train_basic_features <- create_basic_features(train_data)

# Apply same transformations to test data (using training data parameters)
test_basic_features <- create_basic_features(test_data)

write_csv(train_basic_features, "data/processed/train_basic_features.csv")
write_csv(test_basic_features, "data/processed/test_basic_features.csv")

# 5 ANOVA Analysis on Training Data Only ####

## 3.1 Core variables tests ####
smoker_test <- aov(charges ~ smoker, data = train_basic_features)
sex_test <- aov(charges ~ sex, data = train_basic_features)
age_test <- aov(charges ~ age_group_standard, data = train_basic_features)
bmi_test <- aov(charges ~ bmi_category, data = train_basic_features)
region_test <- aov(charges ~ region, data = train_basic_features)
child_test <- aov(charges ~ has_children, data = train_basic_features)

## 3.2 Core variable interactions tests ####
smoker_sex_test <- aov(charges ~ smoker * sex, data = train_basic_features)
smoker_age_test <- aov(charges ~ smoker * age_group_standard, data = train_basic_features)
sex_age_test <- aov(charges ~ sex * age_group_standard, data = train_basic_features)
bmi_smoker_test <- aov(charges ~ bmi_category * smoker, data = train_basic_features)
bmi_sex_test <- aov(charges ~ bmi_category * sex, data = train_basic_features)
sex_child_test <- aov(charges ~ has_children * sex, data = train_basic_features)

## 3.3 Extracting ANOVA results ####

# Function to extract ANOVA statistics
extract_anova_stats <- function(anova_model, effect_name = NULL) {
  anova_summary <- summary(anova_model)
  anova_table <- anova_summary[[1]]
  
  # For main effects (single factor)
  if (is.null(effect_name)) {
    effect_row <- 1  # First row is always the main effect
  } else {
    # For interactions, find the specific effect row
    effect_row <- which(rownames(anova_table) == effect_name)
    if (length(effect_row) == 0) {
      effect_row <- 1  # Default to first row if not found
    }
  }
  
  f_value <- round(anova_table[effect_row, "F value"], 3)
  p_value <- round(anova_table[effect_row, "Pr(>F)"], 3)
  df <- anova_table[effect_row, "Df"]
  sum_sq <- anova_table[effect_row, "Sum Sq"]
  mean_sq <- anova_table[effect_row, "Mean Sq"]
  
  # Calculate effect size (eta-squared)
  total_sum_sq <- sum(anova_table[, "Sum Sq"])
  eta_squared <- round(sum_sq / total_sum_sq, 3)
  
  # Determine significance level
  sig_level <- ifelse(p_value < 0.001, "***",
                      ifelse(p_value < 0.01, "**",
                             ifelse(p_value < 0.05, "*",
                                    ifelse(p_value < 0.1, ".", ""))))
  
  return(list(
    f_value = f_value,
    p_value = p_value,
    df = df,
    sum_sq = sum_sq,
    mean_sq = mean_sq,
    eta_squared = eta_squared,
    sig_level = sig_level,
    significant = p_value < 0.05
  ))
}

# Extract results for main effects
smoker_results <- extract_anova_stats(smoker_test)
sex_results <- extract_anova_stats(sex_test)
age_results <- extract_anova_stats(age_test)
bmi_results <- extract_anova_stats(bmi_test)
region_results <- extract_anova_stats(region_test)
child_results <- extract_anova_stats(child_test)

# Extract results for interactions
smoker_sex_results <- extract_anova_stats(smoker_sex_test, "smoker:sex")
smoker_age_results <- extract_anova_stats(smoker_age_test, "smoker:age_group_standard")
sex_age_results <- extract_anova_stats(sex_age_test, "sex:age_group_standard")
bmi_smoker_results <- extract_anova_stats(bmi_smoker_test, "bmi_category:smoker")
bmi_sex_results <- extract_anova_stats(bmi_sex_test, "bmi_category:sex")
sex_child_results <- extract_anova_stats(sex_child_test, "has_children:sex")

# Results Table
original_anova_results <- tibble(
  analysis_id = 1:12,
  analysis_type = c(rep("One-way", 6), rep("Two-way", 6)),
  variables = c(
    "smoker → charges", "sex → charges", "age_group_standard → charges",
    "bmi_category → charges", "region → charges", "has_children → charges",
    "smoker × sex → charges", "smoker × age_group → charges", 
    "sex × age_group → charges", "bmi_category × smoker → charges",
    "bmi_category × sex → charges", "has_children × sex → charges"
  ),
  f_value = c(
    smoker_results$f_value, sex_results$f_value, age_results$f_value,
    bmi_results$f_value, region_results$f_value, child_results$f_value,
    smoker_sex_results$f_value, smoker_age_results$f_value, sex_age_results$f_value,
    bmi_smoker_results$f_value, bmi_sex_results$f_value, sex_child_results$f_value
  ),
  p_value = c(
    smoker_results$p_value, sex_results$p_value, age_results$p_value,
    bmi_results$p_value, region_results$p_value, child_results$p_value,
    smoker_sex_results$p_value, smoker_age_results$p_value, sex_age_results$p_value,
    bmi_smoker_results$p_value, bmi_sex_results$p_value, sex_child_results$p_value
  ),
  eta_squared = c(
    smoker_results$eta_squared, sex_results$eta_squared, age_results$eta_squared,
    bmi_results$eta_squared, region_results$eta_squared, child_results$eta_squared,
    smoker_sex_results$eta_squared, smoker_age_results$eta_squared, sex_age_results$eta_squared,
    bmi_smoker_results$eta_squared, bmi_sex_results$eta_squared, sex_child_results$eta_squared
  ),
  significant = c(
    smoker_results$significant, sex_results$significant, age_results$significant,
    bmi_results$significant, region_results$significant, child_results$significant,
    smoker_sex_results$significant, smoker_age_results$significant, sex_age_results$significant,
    bmi_smoker_results$significant, bmi_sex_results$significant, sex_child_results$significant
  ),
  effect_size_interpretation = case_when(
    eta_squared < 0.01 ~ "Small",
    eta_squared < 0.06 ~ "Medium", 
    eta_squared < 0.14 ~ "Large",
    TRUE ~ "Very Large"
  )
)

write_csv(original_anova_results, "outputs/tables/original_anova_analysis_results.csv")

# 5 Test Engineered Features (Training Data Only) ####

perform_feature_anova_robust <- function(data, feature_names, target_col = "charges") {
  
  anova_results <- tibble(
    feature_name = character(),
    test_type = character(),
    f_value = numeric(),
    p_value = numeric(),
    eta_squared = numeric(),
    significant = logical(),
    error_message = character()
  )
  
  for (feature in feature_names) {
    
    # Skip if feature doesn't exist
    if (!feature %in% colnames(data)) {
      result <- tibble(
        feature_name = feature,
        test_type = "MISSING",
        f_value = NA_real_,
        p_value = NA_real_,
        eta_squared = NA_real_,
        significant = NA,
        error_message = "Feature not found in dataset"
      )
      anova_results <- bind_rows(anova_results, result)
      next
    }
    
    feature_data <- data[[feature]]
    target_data <- data[[target_col]]
    
    # Remove NA values
    complete_cases <- complete.cases(feature_data, target_data)
    if (sum(complete_cases) < 10) {
      result <- tibble(
        feature_name = feature,
        test_type = "INSUFFICIENT_DATA",
        f_value = NA_real_,
        p_value = NA_real_,
        eta_squared = NA_real_,
        significant = NA,
        error_message = "Insufficient complete cases"
      )
      anova_results <- bind_rows(anova_results, result)
      next
    }
    
    feature_clean <- feature_data[complete_cases]
    target_clean <- target_data[complete_cases]
    
    # Check feature variance
    if (length(unique(feature_clean)) <= 1) {
      result <- tibble(
        feature_name = feature,
        test_type = "NO_VARIANCE",
        f_value = NA_real_,
        p_value = NA_real_,
        eta_squared = NA_real_,
        significant = NA,
        error_message = "Feature has no variance"
      )
      anova_results <- bind_rows(anova_results, result)
      next
    }
    
    # Determine test type and perform analysis
    tryCatch({
      if (is.numeric(feature_clean) && length(unique(feature_clean)) > 10) {
        # Linear regression for continuous
        model <- lm(target_clean ~ feature_clean)
        anova_result <- anova(model)
        model_summary <- summary(model)
        
        f_value <- anova_result$`F value`[1]
        p_value <- anova_result$`Pr(>F)`[1]
        eta_squared <- model_summary$r.squared
        test_type <- "Linear_Regression"
        
      } else {
        # ANOVA for categorical or discrete
        if (is.numeric(feature_clean)) {
          feature_clean <- as.factor(feature_clean)
        }
        
        model <- aov(target_clean ~ feature_clean)
        anova_summary <- summary(model)
        
        f_value <- anova_summary[[1]]$`F value`[1]
        p_value <- anova_summary[[1]]$`Pr(>F)`[1]
        
        # Calculate eta-squared (measure of effect size [.01 = small, .06 = med, .14 = large])
        ss_between <- anova_summary[[1]]$`Sum Sq`[1]
        ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
        eta_squared <- ss_between / ss_total
        test_type <- "ANOVA"
      }
      
      result <- tibble(
        feature_name = feature,
        test_type = test_type,
        f_value = f_value,
        p_value = p_value,
        eta_squared = eta_squared,
        significant = p_value < 0.05,
        error_message = "Success"
      )
      
    }, error = function(e) {
      result <- tibble(
        feature_name = feature,
        test_type = "ERROR",
        f_value = NA_real_,
        p_value = NA_real_,
        eta_squared = NA_real_,
        significant = NA,
        error_message = as.character(e$message)
      )
    })
    
    anova_results <- bind_rows(anova_results, result)
  }
  
  return(anova_results)
}

# Test all created features on training data only
created_features <- c(
  "smoker_encoded", "sex_encoded", "age_squared", "bmi_squared", "age_cubed",
  "age_log", "bmi_log", "has_children", "children_squared", "smoker_age_interaction",
  "smoker_bmi_interaction", "age_bmi_interaction", "smoker_children_interaction",
  "region_southeast", "region_southwest", "region_northwest", "high_bmi_smoker",
  "senior_smoker", "age_group_ordinal", "bmi_category_ordinal"
)

engineered_anova_results <- perform_feature_anova_robust(train_basic_features, created_features)

write_csv(engineered_anova_results, "outputs/tables/engineered_features_anova_results.csv")

# 6 Advanced Feature Engineering (Training Data Only) ####

create_advanced_features <- function(data) {
  data %>%
    mutate(
      # Smoking Duration Impact Modeling (Literature-Based)
      smoker_age_severity_index = case_when(
        smoker == "no" ~ 0,
        smoker == "yes" & age < 30 ~ age * 0.8,
        smoker == "yes" & age < 45 ~ age * 1.2,
        smoker == "yes" & age < 60 ~ age * 1.8,
        smoker == "yes" & age >= 60 ~ age * 2.5
      ),
      
      # Metabolic Syndrome Risk (BMI + Age Compound Effects)
      metabolic_syndrome_risk = case_when(
        bmi < 25 & age < 40 ~ 1.0,
        bmi >= 25 & bmi < 30 & age < 40 ~ 1.3,
        bmi >= 30 & age < 40 ~ 1.8,
        bmi < 25 & age >= 40 ~ 1.2,
        bmi >= 25 & bmi < 30 & age >= 40 & age < 55 ~ 1.6,
        bmi >= 30 & age >= 40 & age < 55 ~ 2.2,
        bmi >= 25 & age >= 55 ~ 2.8,
        bmi >= 30 & age >= 55 ~ 3.5,
        TRUE ~ 1.0
      ),
      
      # Family Cost Optimization (Economies of Scale)
      family_cost_optimization = case_when(
        children == 0 ~ 1.0,
        children == 1 ~ 0.85,
        children == 2 ~ 0.75,
        children == 3 ~ 0.70,
        children >= 4 ~ 0.65,
        TRUE ~ 1.0
      ),
      
      # Compound Lifestyle Risk Score (Smoking + BMI)
      compound_lifestyle_risk = (
        ifelse(smoker == "yes", 2.5, 1.0) *
          case_when(
            bmi < 18.5 ~ 1.3,
            bmi >= 18.5 & bmi < 25 ~ 1.0,
            bmi >= 25 & bmi < 30 ~ 1.2,
            bmi >= 30 ~ 1.5,
            TRUE ~ 1.0
          ) *
          case_when(
            age < 30 ~ 0.8,
            age >= 30 & age < 50 ~ 1.0,
            age >= 50 ~ 1.3,
            TRUE ~ 1.0
          )
      )
    )
}

# Apply advanced features to training data
train_advanced <- train_basic_features %>%
  create_advanced_features()

# Apply same transformations to test data
test_advanced <- test_basic_features %>%
  create_advanced_features()

write_csv(train_advanced, "data/processed/train_advanced_features.csv")
write_csv(test_advanced, "data/processed/test_advanced_features.csv")

# Test advanced features on training data only
new_advanced_features <- c(
  "smoker_age_severity_index", "metabolic_syndrome_risk", 
  "family_cost_optimization", "compound_lifestyle_risk"
)

advanced_feature_results <- perform_feature_anova_robust(train_advanced, new_advanced_features)

write_csv(advanced_feature_results, "outputs/tables/advanced_features_anova_results.csv")

# 7 Final Feature Set Preparation ####

create_final_encoding <- function(data) {
  data %>%
    mutate(
      # Binary encodings
      is_smoker = as.numeric(smoker == "yes"),
      is_male = as.numeric(sex == "male"),
      is_obese = as.numeric(bmi_category == "obese"),
      has_children_flag = as.numeric(children > 0),
      
      # Interactions
      smoker_male = is_smoker * is_male,
      smoker_obese = is_smoker * is_obese,
      high_risk_combo = is_smoker * is_obese * is_male,
      
      # Ordinal encodings
      bmi_risk_ordinal = case_when(
        bmi_category == "underweight" ~ 1,
        bmi_category == "normal" ~ 2,
        bmi_category == "overweight" ~ 3,
        bmi_category == "obese" ~ 4,
        TRUE ~ 2
      ),
      
      region_cost_ordinal = case_when(
        region == "southeast" ~ 1,
        region == "southwest" ~ 2,
        region == "northwest" ~ 3,
        region == "northeast" ~ 4,
        TRUE ~ 2
      ),
      
      # Age group binary flags
      is_young_adult = as.numeric(age_group_standard == "young_adult"),
      is_senior = as.numeric(age_group_standard == "senior"),
      is_middle_aged = as.numeric(age_group_standard %in% c("middle_aged", "pre_senior")),
      
      # Region dummies
      region_northeast = as.numeric(region == "northeast")
      # region_northwest, region_southeast, region_southwest already created
    )
}

# Apply final encoding to both datasets
train_encoded <- train_advanced %>%
  create_final_encoding()

test_encoded <- test_advanced %>%
  create_final_encoding()

write_csv(train_encoded, "data/processed/train_encoded_final.csv")
write_csv(test_encoded, "data/processed/test_encoded_final.csv")

# 8 Feature Selection Summary ####

# All ANOVA results for comprehensive view (training data only)
complete_feature_analysis <- bind_rows(
  original_anova_results %>%
    select(variables, f_value, p_value, eta_squared, significant, effect_size_interpretation) %>%
    rename(feature_name = variables) %>%
    mutate(analysis_source = "Original_Variables"),
  
  engineered_anova_results %>%
    filter(error_message == "Success") %>%
    select(feature_name, f_value, p_value, eta_squared, significant) %>%
    mutate(
      effect_size_interpretation = case_when(
        eta_squared >= 0.14 ~ "Large",
        eta_squared >= 0.06 ~ "Medium",
        eta_squared >= 0.01 ~ "Small",
        TRUE ~ "Negligible"
      ),
      analysis_source = "Engineered_Features"
    ),
  
  advanced_feature_results %>%
    filter(error_message == "Success") %>%
    select(feature_name, f_value, p_value, eta_squared, significant) %>%
    mutate(
      effect_size_interpretation = case_when(
        eta_squared >= 0.14 ~ "Large",
        eta_squared >= 0.06 ~ "Medium",
        eta_squared >= 0.01 ~ "Small",
        TRUE ~ "Negligible"
      ),
      analysis_source = "Advanced_Features"
    )
) %>%
  arrange(desc(eta_squared))

write_csv(complete_feature_analysis, "outputs/tables/complete_feature_analysis_final.csv")

# Top performing features summary
top_features <- complete_feature_analysis %>%
  filter(significant == TRUE) %>%
  slice_head(n = 20) %>%
  select(feature_name, eta_squared, effect_size_interpretation, analysis_source)

write_csv(top_features, "outputs/tables/top_performing_features.csv")

# 9 Data Dictionary Update ####

# Function to bulk add engineered features to dictionary
add_engineered_features_to_dictionary <- function() {
  
  # Basic Features
  add_to_dictionary("smoker_encoded", "numeric", "Script 2 - Basic Features", 
                    "Binary encoding: 1 for smoker, 0 for non-smoker", "0, 1")
  
  add_to_dictionary("sex_encoded", "numeric", "Script 2 - Basic Features",
                    "Binary encoding: 1 for male, 0 for female", "0, 1")
  
  add_to_dictionary("age_group_standard", "character", "Script 2 - Basic Features", 
                    "Standardized age groups", "young_adult, adult, middle_aged, pre_senior, senior")
  
  add_to_dictionary("bmi_category", "character", "Script 2 - Basic Features",
                    "BMI classification categories", "underweight, normal, overweight, obese")
  
  add_to_dictionary("age_squared", "numeric", "Script 2 - Basic Features",
                    "Age squared to capture non-linear age effects", "324 to 4225")
  
  add_to_dictionary("bmi_squared", "numeric", "Script 2 - Basic Features", 
                    "BMI squared to capture non-linear BMI effects", "225 to 2500+")
  
  add_to_dictionary("age_cubed", "numeric", "Script 2 - Basic Features",
                    "Age cubed for extreme non-linear age effects", "5832 to 274625")
  
  add_to_dictionary("age_log", "numeric", "Script 2 - Basic Features",
                    "Natural logarithm of age", "2.89 to 4.19")
  
  add_to_dictionary("bmi_log", "numeric", "Script 2 - Basic Features",
                    "Natural logarithm of BMI", "2.71 to 3.91")
  
  add_to_dictionary("has_children", "numeric", "Script 2 - Basic Features",
                    "Binary indicator for having children", "0, 1")
  
  add_to_dictionary("children_squared", "numeric", "Script 2 - Basic Features",
                    "Children count squared", "0 to 25")
  
  add_to_dictionary("smoker_age_interaction", "numeric", "Script 2 - Basic Features",
                    "Product of smoker encoding and age", "0 to 65")
  
  add_to_dictionary("smoker_bmi_interaction", "numeric", "Script 2 - Basic Features",
                    "Product of smoker encoding and BMI", "0 to 53.13")
  
  add_to_dictionary("age_bmi_interaction", "numeric", "Script 2 - Basic Features",
                    "Product of age and BMI", "270 to 3445")
  
  add_to_dictionary("smoker_children_interaction", "numeric", "Script 2 - Basic Features",
                    "Product of smoker encoding and has_children", "0, 1")
  
  add_to_dictionary("region_southeast", "numeric", "Script 2 - Basic Features",
                    "Binary flag for southeast region", "0, 1")
  
  add_to_dictionary("region_southwest", "numeric", "Script 2 - Basic Features",
                    "Binary flag for southwest region", "0, 1")
  
  add_to_dictionary("region_northwest", "numeric", "Script 2 - Basic Features",
                    "Binary flag for northwest region", "0, 1")
  
  add_to_dictionary("age_bins", "factor", "Script 2 - Basic Features",
                    "Age grouped into 5-year bins", "(15,20], (20,25], ..., (65,70]")
  
  add_to_dictionary("bmi_health_category", "character", "Script 2 - Basic Features",
                    "Health categories based on BMI thresholds", "underweight, normal, overweight, obese")
  
  add_to_dictionary("high_bmi_smoker", "numeric", "Script 2 - Basic Features",
                    "Binary flag for obese smokers", "0, 1")
  
  add_to_dictionary("senior_smoker", "numeric", "Script 2 - Basic Features",
                    "Binary flag for senior smokers", "0, 1")
  
  add_to_dictionary("age_group_ordinal", "numeric", "Script 2 - Basic Features",
                    "Ordinal encoding of age groups", "1, 2, 3, 4, 5")
  
  add_to_dictionary("bmi_category_ordinal", "numeric", "Script 2 - Basic Features",
                    "Ordinal encoding of BMI categories", "1, 2, 3, 4")
  
  # Advanced Features
  add_to_dictionary("smoker_age_severity_index", "numeric", "Script 2 - Advanced Features",
                    "Age-adjusted smoking severity with literature-based multipliers", "0 to 162.5")
  
  add_to_dictionary("metabolic_syndrome_risk", "numeric", "Script 2 - Advanced Features",
                    "Compound risk score for metabolic syndrome based on BMI and age", "1.0 to 3.5")
  
  add_to_dictionary("family_cost_optimization", "numeric", "Script 2 - Advanced Features",
                    "Family size economies of scale factor", "0.65 to 1.0")
  
  add_to_dictionary("compound_lifestyle_risk", "numeric", "Script 2 - Advanced Features",
                    "Comprehensive lifestyle risk: smoking * BMI_risk * age_risk", "0.8 to 4.875")
  
  # Final Encoding Features
  add_to_dictionary("is_smoker", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for smoking status", "0, 1")
  
  add_to_dictionary("is_male", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for male sex", "0, 1")
  
  add_to_dictionary("is_obese", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for obesity (BMI ≥30)", "0, 1")
  
  add_to_dictionary("has_children_flag", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for having any children", "0, 1")
  
  add_to_dictionary("smoker_male", "numeric", "Script 2 - Final Encoding",
                    "Interaction: is_smoker * is_male", "0, 1")
  
  add_to_dictionary("smoker_obese", "numeric", "Script 2 - Final Encoding",
                    "Interaction: is_smoker * is_obese", "0, 1")
  
  add_to_dictionary("high_risk_combo", "numeric", "Script 2 - Final Encoding",
                    "Triple interaction: is_smoker * is_obese * is_male", "0, 1")
  
  add_to_dictionary("bmi_risk_ordinal", "numeric", "Script 2 - Final Encoding",
                    "Ordinal encoding of BMI categories", "1, 2, 3, 4")
  
  add_to_dictionary("region_cost_ordinal", "numeric", "Script 2 - Final Encoding",
                    "Ordinal encoding of regions by cost", "1, 2, 3, 4")
  
  add_to_dictionary("is_young_adult", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for young adult age group", "0, 1")
  
  add_to_dictionary("is_senior", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for senior age group", "0, 1")
  
  add_to_dictionary("is_middle_aged", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for middle-aged groups", "0, 1")
  
  add_to_dictionary("region_northeast", "numeric", "Script 2 - Final Encoding",
                    "Binary flag for northeast region", "0, 1")
  
  # Benchmark data
  add_to_dictionary("avg_hcup_charges", "numeric", "Script 2 - External Benchmarking",
                    "Average hospital charges by age group from HCUP data", "National benchmark values")
}

# Execute the dictionary updates
add_engineered_features_to_dictionary()

# Function to add ANOVA results as metadata
add_anova_metadata_to_dictionary <- function() {
  if(file.exists("outputs/tables/complete_feature_analysis_final.csv")) {
    anova_results <- read_csv("outputs/tables/complete_feature_analysis_final.csv", show_col_types = FALSE)
    
    for(i in 1:nrow(anova_results)) {
      feature_name <- anova_results$feature_name[i]
      eta_squared <- round(anova_results$eta_squared[i], 3)
      p_value <- round(anova_results$p_value[i], 3)
      
      anova_note <- paste0("ANOVA: η²=", eta_squared, ", p=", p_value)
      
      tryCatch({
        update_dictionary(feature_name, notes = anova_note)
      }, error = function(e) {
        # Feature might not exist in dictionary yet
      })
    }
  }
}

add_anova_metadata_to_dictionary()

# Final data dictionary
write_csv(data_dictionary, "data/data_dictionary.csv")

# Summary statistics
dict_summary <- data_dictionary %>%
  group_by(source) %>%
  summarise(
    n_variables = n(),
    n_numeric = sum(data_type %in% c("numeric", "integer")),
    n_categorical = sum(data_type %in% c("character", "factor")),
    .groups = "drop"
  ) %>%
  arrange(desc(n_variables))

write_csv(dict_summary, "outputs/tables/data_dictionary_summary.csv")

# 10 Scaling Preparation (Training Data Parameters Only) ####

prepare_scaling_parameters <- function(train_data) {
  # Identify numeric columns for scaling (exclude target)
  numeric_cols <- train_data %>%
    select_if(is.numeric) %>%
    select(-charges) %>%
    colnames()
  
  # Remove zero variance columns
  zero_variance_cols <- numeric_cols[sapply(numeric_cols, function(col) {
    var(train_data[[col]], na.rm = TRUE) == 0
  })]
  
  if (length(zero_variance_cols) > 0) {
    numeric_cols <- setdiff(numeric_cols, zero_variance_cols)
  }
  
  # Calculate scaling parameters from training data only
  scaling_params <- tibble(
    feature = numeric_cols,
    mean_val = sapply(numeric_cols, function(x) mean(train_data[[x]], na.rm = TRUE)),
    sd_val = sapply(numeric_cols, function(x) sd(train_data[[x]], na.rm = TRUE)),
    min_val = sapply(numeric_cols, function(x) min(train_data[[x]], na.rm = TRUE)),
    max_val = sapply(numeric_cols, function(x) max(train_data[[x]], na.rm = TRUE)),
    range_val = max_val - min_val,
    cv = abs(sd_val / mean_val),
    
    needs_scaling = range_val > 10 | cv > 1,
    scaling_method = case_when(
      range_val > 1000 ~ "normalize",
      cv > 2 ~ "standardize", 
      range_val > 10 ~ "normalize",
      TRUE ~ "none"
    )
  )
  
  return(list(
    scaling_params = scaling_params,
    numeric_features = numeric_cols,
    zero_variance_features = zero_variance_cols
  ))
}

# Apply scaling function to create scaling parameters from training data
scaling_prep <- prepare_scaling_parameters(train_encoded)

# Function to apply scaling using training parameters
apply_scaling <- function(data, scaling_params) {
  scaled_data <- data
  
  for(i in 1:nrow(scaling_params$scaling_params)) {
    feature <- scaling_params$scaling_params$feature[i]
    method <- scaling_params$scaling_params$scaling_method[i]
    
    if(feature %in% names(data) && method != "none") {
      if(method == "standardize") {
        mean_val <- scaling_params$scaling_params$mean_val[i]
        sd_val <- scaling_params$scaling_params$sd_val[i]
        scaled_data[[feature]] <- (data[[feature]] - mean_val) / sd_val
      } else if(method == "normalize") {
        min_val <- scaling_params$scaling_params$min_val[i]
        max_val <- scaling_params$scaling_params$max_val[i]
        scaled_data[[feature]] <- (data[[feature]] - min_val) / (max_val - min_val)
      }
    }
  }
  
  # Remove zero variance features identified from training
  if(length(scaling_params$zero_variance_features) > 0) {
    scaled_data <- scaled_data %>% 
      select(-any_of(scaling_params$zero_variance_features))
  }
  
  return(scaled_data)
}

# Apply scaling to both datasets using training parameters
train_scaled <- apply_scaling(train_encoded, scaling_prep)
test_scaled <- apply_scaling(test_encoded, scaling_prep)

write_csv(scaling_prep$scaling_params, "outputs/tables/scaling_parameters.csv")
write_csv(train_scaled, "data/processed/train_scaled_final.csv")
write_csv(test_scaled, "data/processed/test_scaled_final.csv")

# 11 Feature Selection for Modeling ####

# High-impact feature set (Large + Medium effect sizes)
if (exists("complete_feature_analysis") && nrow(complete_feature_analysis) > 0) {
  high_impact_features <- complete_feature_analysis %>%
    filter(significant == TRUE, eta_squared >= 0.06) %>%
    pull(feature_name)
  
  essential_features <- complete_feature_analysis %>%
    filter(significant == TRUE, eta_squared >= 0.14) %>%
    pull(feature_name)
} else {
  high_impact_features <- c("smoker_encoded", "smoker_age_interaction", 
                            "compound_lifestyle_risk", "age_squared")
  essential_features <- c("smoker_encoded", "age_squared")
}

# Feature sets that exist in the data
available_high_impact <- high_impact_features[high_impact_features %in% colnames(train_encoded)]
available_essential <- essential_features[essential_features %in% colnames(train_encoded)]

# Create modeling datasets
if (length(available_high_impact) > 0) {
  train_high_impact <- train_encoded %>%
    select(all_of(c("charges", available_high_impact)))
  test_high_impact <- test_encoded %>%
    select(all_of(c("charges", available_high_impact)))
  
  train_high_impact_scaled <- train_scaled %>%
    select(all_of(c("charges", available_high_impact[available_high_impact %in% names(train_scaled)])))
  test_high_impact_scaled <- test_scaled %>%
    select(all_of(c("charges", available_high_impact[available_high_impact %in% names(test_scaled)])))
} else {
  train_high_impact <- train_encoded %>%
    select(charges, smoker_encoded, compound_lifestyle_risk, age_squared)
  test_high_impact <- test_encoded %>%
    select(charges, smoker_encoded, compound_lifestyle_risk, age_squared)
  
  train_high_impact_scaled <- train_scaled %>%
    select(charges, smoker_encoded, compound_lifestyle_risk, age_squared)
  test_high_impact_scaled <- test_scaled %>%
    select(charges, smoker_encoded, compound_lifestyle_risk, age_squared)
}

if (length(available_essential) > 0) {
  train_essential <- train_encoded %>%
    select(all_of(c("charges", available_essential)))
  test_essential <- test_encoded %>%
    select(all_of(c("charges", available_essential)))
  
  train_essential_scaled <- train_scaled %>%
    select(all_of(c("charges", available_essential[available_essential %in% names(train_scaled)])))
  test_essential_scaled <- test_scaled %>%
    select(all_of(c("charges", available_essential[available_essential %in% names(test_scaled)])))
} else {
  train_essential <- train_encoded %>%
    select(charges, smoker_encoded, age_squared)
  test_essential <- test_encoded %>%
    select(charges, smoker_encoded, age_squared)
  
  train_essential_scaled <- train_scaled %>%
    select(charges, smoker_encoded, age_squared)
  test_essential_scaled <- test_scaled %>%
    select(charges, smoker_encoded, age_squared)
}

# Save all modeling datasets
write_csv(train_high_impact, "data/processed/train_high_impact.csv")
write_csv(test_high_impact, "data/processed/test_high_impact.csv")
write_csv(train_high_impact_scaled, "data/processed/train_high_impact_scaled.csv")
write_csv(test_high_impact_scaled, "data/processed/test_high_impact_scaled.csv")

write_csv(train_essential, "data/processed/train_essential.csv")
write_csv(test_essential, "data/processed/test_essential.csv")
write_csv(train_essential_scaled, "data/processed/train_essential_scaled.csv")
write_csv(test_essential_scaled, "data/processed/test_essential_scaled.csv")

# Feature selection summary
feature_selection_summary <- tibble(
  dataset = c("High Impact", "High Impact Scaled", "Essential", "Essential Scaled"),
  n_features = c(
    ncol(train_high_impact) - 1,
    ncol(train_high_impact_scaled) - 1,
    ncol(train_essential) - 1,
    ncol(train_essential_scaled) - 1
  ),
  n_observations_train = c(
    nrow(train_high_impact),
    nrow(train_high_impact_scaled),
    nrow(train_essential),
    nrow(train_essential_scaled)
  ),
  n_observations_test = c(
    nrow(test_high_impact),
    nrow(test_high_impact_scaled),
    nrow(test_essential),
    nrow(test_essential_scaled)
  ),
  selection_criteria = c(
    "eta_squared >= 0.06 (Medium+ effect)",
    "eta_squared >= 0.06 (Medium+ effect) - Scaled",
    "eta_squared >= 0.14 (Large effect)",
    "eta_squared >= 0.14 (Large effect) - Scaled"
  )
)

write_csv(feature_selection_summary, "outputs/tables/feature_selection_summary.csv")

# 12 Final Summary ####

# Create final summary of what was produced
final_summary <- tibble(
  Stage = c("Data Split", "Basic Features", "Advanced Features", "Final Encoding", 
            "Feature Analysis", "Scaling", "Model-Ready Datasets"),
  Description = c(
    "70/30 train/test split with stratification",
    "Domain-based feature engineering on training data",
    "Complex risk modeling features",
    "Final binary/ordinal encodings", 
    "ANOVA analysis for feature selection",
    "Standardization/normalization using training parameters",
    "High-impact and essential feature sets ready for modeling"
  ),
  Files_Created = c(
    "train_basic_features.csv, test_basic_features.csv",
    "train_advanced_features.csv, test_advanced_features.csv", 
    "train_encoded_final.csv, test_encoded_final.csv",
    "train_scaled_final.csv, test_scaled_final.csv",
    "complete_feature_analysis_final.csv, top_performing_features.csv",
    "scaling_parameters.csv",
    "train_high_impact*.csv, test_high_impact*.csv, train_essential*.csv, test_essential*.csv"
  )
)

write_csv(final_summary, "outputs/tables/script_2_final_summary.csv")