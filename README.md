# Healthcare Cost Prediction Model 🏥

A comprehensive machine learning system for healthcare insurance cost prediction, featuring advanced statistical analysis, rigorous model evaluation, and systematic algorithm comparison.

## 🎯 Project Overview

This project develops and validates predictive models for healthcare insurance costs using the Kaggle Medical Cost Personal Dataset. The analysis follows industry best practices from exploratory data analysis through comprehensive model evaluation, featuring statistical validation and business impact analysis.

### Key Features
- **Evidence-Based Feature Engineering**: ANOVA-guided feature selection with statistical validation
- **External Data Integration**: HCUP hospital cost benchmarks for population-level validation
- **Advanced Statistical Analysis**: Comprehensive hypothesis testing with effect size measurements
- **Business-Focused Insights**: ROI analysis, intervention targeting, and actionable recommendations
- **Reproducible Pipeline**: Systematic script progression with validation checkpoints

### Key Achievements

- **Systematic Model Comparison**: 8 algorithms rigorously evaluated with cross-validation
- **Best Performance**: Lasso regression achieving 84.6% R² with $4,933 RMSE on validation data
- **Comprehensive Feature Engineering**: 32 engineered features with statistical validation
- **Business Impact Analysis**: ROI calculations and pricing strategy recommendations
- **Production Pipeline**: Complete workflow from data prep to deployment readiness

## 📊 Model Performance Results

### Algorithm Comparison
| Model | RMSE | R² | Type | Rank | MAE |
|-------|------|----|----|------|-----|
| **Lasso** | $4,933 | **84.6%** | Regularized | 1st | $2,531 |
| **Stepwise** | $4,937 | **84.6%** | Linear | 2nd | $2,570 |
| **Elastic Net** | $4,943 | **84.5%** | Regularized | 3rd | $2,533 |
| **Full Model** | $4,965 | **84.4%** | Linear | 4th | $2,548 |
| **Ridge** | $5,011 | **84.1%** | Regularized | 5th | $2,720 |
| **Random Forest** | $5,261 | **80.4%** | Tree-based | 6th | $2,912 |
| **GBM** | $5,335 | **79.9%** | Tree-based | 7th | $2,916 |
| **Simple (Top 5)** | $5,337 | **81.9%** | Linear | 8th | $3,302 |

### Cross-Validation Performance
**Top Performing Models (CV Results):**
- **Random Forest**: Mean RMSE $4,401 (Range: $3,675 - $4,855)
- **GBM**: Mean RMSE $4,354 (Range: $4,024 - $4,491)

*Note: Cross-validation shows tree-based models performing better than single holdout validation*

## 📊 Dataset Information

### Primary Dataset
- **Source**: [Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size**: 1,338 observations, 7 core variables
- **Target**: Insurance charges (continuous, USD)
- **Features**: Age, sex, BMI, children, smoker status, region

### External Benchmarking Data
- **Source**: HCUP (Healthcare Cost and Utilization Project)
- **Purpose**: National hospital cost benchmarks by age group
- **Integration**: Cost ratio features and validation metrics

### Engineered Features
- **32 Engineered Variables**: Mathematically derived, interaction terms, risk scores
- **Statistical Validation**: ANOVA-tested with effect size measurements (η²)
- **Business Intelligence**: Cost efficiency, intervention priority, market opportunity features

## 🔬 Methodology

This project follows an adapted CRISP-DM methodology:

1. **Data Understanding & Exploration** ✅
2. **Data Cleaning & Preparation** ✅  
3. **Feature Engineering & Selection** ✅
4. **Model Development & Training** ✅
5. **Model Evaluation & Validation** ✅
6. **Business Insights & Deployment** ✅

## 📈 Model Selection Analysis

### Selection Criteria
- **Performance**: Lasso achieved lowest RMSE and highest R² on validation set
- **Interpretability**: Linear model provides clear coefficient interpretation
- **Regularization**: L1 penalty provides automatic feature selection
- **Efficiency**: High performance-to-complexity ratio
- **Business Value**: Strong interpretability for stakeholder communication

### Efficiency Analysis Results
| Model | Performance Rank | Complexity Score | Interpretability | Efficiency Score |
|-------|------------------|------------------|------------------|-----------------|
| **Stepwise** | 2nd | 2 | 9/10 | **4.5** |
| **Simple (Top 5)** | 8th | 1 | 10/10 | 3.0 |
| **Lasso** | 1st | 4 | 8/10 | **2.5** |

## 🔧 Advanced Feature Engineering

### Statistical Validation Results
From comprehensive ANOVA analysis of 32 engineered features:

- **Large Effect Features (η² > 0.6)**: 6 features achieving exceptional predictive power
- **Medium Effect Features (η² 0.06-0.10)**: 6 features providing solid contributions  
- **Small Effect Features (η² 0.03-0.06)**: 5 features adding precision
- **Statistical Significance**: 29/32 features significant at p < 0.05

### Key Engineering Innovations
- **Interaction Modeling**: Smoker × BMI, Smoker × Age capturing compound risks
- **Risk Scoring**: Multi-dimensional health and demographic risk indices
- **Non-linear Transformations**: Age/BMI squared, cubed, and log transformations
- **Statistical Validation**: ANOVA-guided feature selection with effect size quantification

## 💰 Business Impact

### Precision-Driven Benefits
- **Pricing Accuracy**: 84.6% R² enables competitive premium setting
- **Risk Management**: $4,933 average error provides reliable cost estimates
- **Model Interpretability**: Linear coefficients enable clear business logic
- **Regulatory Compliance**: Transparent model structure supports audit requirements

### Model Validation Insights
- **Consistent Performance**: Cross-validation confirms model stability
- **Algorithm Diversity**: Tree-based and linear methods show different strengths
- **Feature Engineering Value**: Engineered features significantly improve predictions

## 📁 Project Structure

```
healthcare-cost-prediction/
├── scripts/
│   ├── 00_housekeeping.R                   # Project infrastructure & setup
│   ├── 01_data_exploration_cleaning.R      # EDA and data preparation
│   ├── 02_feature_engineering.R            # Advanced feature creation
│   ├── 03_modeling.R                       # Multi-algorithm training
│   ├── 04_model_evaluation.R               # Comprehensive validation
│   └── 05_final_analysis.R                 # Business insights & deployment
├── data/
│   ├── raw/                               # Original Kaggle dataset
│   ├── processed/                         # Production-ready datasets
│   ├── external/                          # HCUP benchmarking data
│   └── data_dictionary.csv               # Variable documentation
├── outputs/
│   ├── tables/
│   │   ├── comprehensive_model_comparison.csv
│   │   ├── final_model_performance.csv
│   │   ├── cross_validation_results.csv
│   │   └── model_efficiency_analysis.csv
│   └── models/
│       └── final_insurance_cost_model.rds
└── tableau/
    └── tableau_data/                      # Dashboard-ready extracts
```

## 🤝 Documentation for Review

### Key Results Files
1. **Model Performance**: `comprehensive_model_comparison.csv`, `final_model_performance.csv`
2. **Cross-Validation**: `cross_validation_results.csv`  
3. **Efficiency Analysis**: `model_efficiency_analysis.csv`
4. **Production Model**: `final_insurance_cost_model.rds`

### Seeking Feedback On
- **Methodology Validation**: Statistical approach and feature engineering effectiveness
- **Model Selection**: Validation of Lasso selection vs. cross-validation results
- **Business Applications**: Additional use cases and deployment strategies
- **Performance Assessment**: Evaluation of 84.6% R² in healthcare cost prediction context

## 📊 Technical Summary

### Performance Highlights
- **84.6% R² accuracy** - Strong predictive performance for insurance cost modeling
- **$4,933 RMSE** - Reasonable prediction error for business applications
- **Comprehensive validation** - Cross-validation, efficiency analysis, multiple algorithms
- **Advanced feature engineering** - 32 engineered features with statistical validation

### Methodology Strengths
- **Systematic comparison** - 8 algorithms evaluated with consistent metrics
- **Statistical rigor** - ANOVA-guided feature selection with effect size quantification
- **Business focus** - Efficiency analysis considering complexity and interpretability
- **Production readiness** - Complete pipeline with deployment considerations

---

*This project demonstrates systematic machine learning methodology, achieving strong predictive performance (84.6% R²) in healthcare cost prediction through rigorous model comparison, advanced feature engineering, and comprehensive validation.*
