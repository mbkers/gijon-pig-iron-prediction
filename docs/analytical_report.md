# Analytical report: Pig iron production forecasting using satellite thermal data

## Summary

This report presents a machine learning approach to predict monthly pig iron production at the Gijón steel mill using thermal signatures from Sentinel-2 satellite imagery. Gaussian Process Regression (GPR) achieves R² = 0.338 with well-calibrated uncertainty estimates, providing valuable insights for supply chain management and operational monitoring despite limited historical data.

**Key results**: The model demonstrates honest predictive capability (LOO-CV R² = 0.338) with 95.8% prediction interval coverage using only 24 months of data, showing that thermal satellite monitoring can provide meaningful production insights even with constrained datasets.

## Project overview

### Business context
Accurate estimation of industrial production is critical for commodity markets and supply chain management. Traditional methods rely on ground-based reporting, which can suffer from delays and inaccuracies. This project explores satellite thermal observations to estimate pig iron production, offering a timely and objective alternative.

### Data sources
- **Satellite data**: Sentinel-2 imagery (2022-2023) with pre-computed thermal indices (TAI, NHI_SWIR, NHI_SWNIR) at 10-20m resolution, totalling ~290 observations with 5-day revisit frequency
- **Ground truth**: Monthly pig iron production data for Spain (24 months, 182-344k tons/month)
- **Spatial data**: Component masks for blast furnaces, coke ovens, sintering plants, and plant perimeter
- **Quality control**: Cloud masks for contamination assessment

### Methodology
The project employs a systematic approach combining cloud-aware feature extraction with Gaussian Process Regression to predict monthly production whilst quantifying uncertainty. Detailed implementation can be found in the accompanying Jupyter notebooks:
- **[00_quickstart.ipynb](notebooks/00_quickstart.ipynb)**: Quick start guide and basic usage examples
- **[01_eda.ipynb](notebooks/01_eda.ipynb)**: Comprehensive exploratory data analysis of production trends and thermal signatures
- **[02_model_results.ipynb](notebooks/02_model_results.ipynb)**: In-depth analysis of model performance and predictions

## Evolution of approach

### Initial exploration: Component-based analysis
**Approach**: Extract thermal signatures using predefined industrial component masks (blast furnaces, coke ovens).
**Outcome**: Failed due to extremely small mask sizes (e.g. 16 pixels for Blast Furnace A).
**Learning**: Pre-defined masks may not capture the actual thermal footprint of industrial processes.

### Hotspot-based feature engineering
**Approach**: Extract statistics from the entire plant perimeter, focusing on extreme values (percentiles, threshold counts).
**Outcome**: Improved results with best correlation of 0.447 (NHI_SWNIR_clear_mean_mean).
**Learning**: Extreme thermal values as well as average temperatures capture industrial heat signatures well.

### Traditional ML exploration
**Approach**: Applied Random Forest and Linear Regression with extensive feature engineering.
**Outcome**: Severe overfitting (Random Forest: Train R²=0.82, Test R²=-0.66).
**Learning**: With only 24 data points, complex models overfit dramatically regardless of methodology quality.

### Final solution: Gaussian Process Regression
**Approach**: Implement GPR with careful feature selection and uncertainty quantification.
**Outcome**: Honest performance (LOO-CV R² = 0.338) with reliable uncertainty estimates (95.8% coverage).
**Learning**: For small samples, simpler models with uncertainty quantification provide more value than complex models with high apparent accuracy.

**Why GPR?**
1. **Built-in uncertainty quantification**: Provides prediction intervals, not just point estimates that also provides robust handling of missing operational context
2. **Small sample friendly**: Bayesian framework works well with limited data (n=24)
3. **Flexible kernels**: Matern kernel captures non-smooth industrial processes
4. **Robust to overfitting**: With proper feature selection
5. **Non-parametric flexibility**: Adapts to data patterns without strong assumptions

## Key challenges and solutions

### Challenge 1: Small sample size (n=24)
**Solution**: Use Leave-One-Out Cross-Validation (appropriate for small sample sizes) for honest assessment, limit to top features, and embrace uncertainty rather than pursuing high R².

### Challenge 2: Temporal misalignment
**Solution**: Convert daily satellite observations to monthly aggregates using maximum values and 95th/99th percentiles for robust extreme value estimation, combined with cloud-aware features.

### Challenge 3: Missing operational context
**Solution**: Accept model limitations (i.e. no data on maintenance, shutdowns, or operational changes) and focus on reliable uncertainty bounds for decision-making, anomaly detection capabilities, and baseline performance superior to naive forecasts.

## Technical approach

### Feature engineering
1. **Thermal indices**: TAI, NHI_SWIR, NHI_SWNIR capture different aspects of industrial heat
2. **Statistical aggregation**: Mean, maximum, percentiles (95th, 99th), standard deviation
3. **Threshold counts**: Pixels above 0.5, 1.0, 1.5, 2.0 thermal units
4. **Cloud-aware processing**: Statistics computed exclusively from clear pixels

### Model architecture
```python
# GPR with Matern kernel
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level)
```

**Kernel interpretation**:
- **Matern(nu=1.5)**: Captures non-smooth but continuous industrial processes
- **WhiteKernel**: Accounts for observation noise (~47% of variance)
- **Length scale ≈ 1**: Features decorrelate over one standard deviation

### Feature selection
Top 5 features by correlation with production:
1. **NHI_SWNIR_clear_mean_mean**: Clear sky thermal averages
2. **NHI_SWNIR_p99_max**: Maximum extreme heat events
3. **TAI_p95_std**: Thermal variability patterns
4. **NHI_SWIR_clear_mean_mean**: SWIR thermal signatures
5. **NHI_SWNIR_clear_p99_mean**: Average extreme thermal activity

## Results and performance

<div align="center">
    <img src="/results/figures/loo_predictions.png" alt="LOO-CV figure">
</div>

### Model performance metrics
- **Leave-One-Out CV**: RMSE = 43.5k tons, R² = 0.338, MAE = 33.3k tons
- **95% prediction interval coverage**: 95.8% (well-calibrated)
- **Mean prediction uncertainty**: ±85 thousand tons

### Key insights
1. **SWIR indices superior**: Short-wave infrared better captures industrial heat than other spectral bands
2. **High noise component**: 47% of variance remains unexplained, indicating missing operational variables
3. **Reliable uncertainty bounds**: Well-calibrated intervals enable informed decision-making
<!-- 2. **Extremes matter**: 99th percentile temperatures more predictive than average values -->

### Practical value
Despite modest R², the model provides:
- **Actionable uncertainty bounds**: "Production likely between X and Y tons"
- **Anomaly detection capability**: Flag when actual production falls outside prediction intervals
- **Baseline forecasting**: Performance superior to naive monthly averages
- **Honest assessment**: LOO-CV prevents overconfident claims

### Model limitations
- Cannot capture sudden operational changes (maintenance schedules, equipment failures)
- Limited to monthly aggregation, losing daily production dynamics
- Performance dependent on clear sky conditions for satellite observations (motivating multi-sensor fusion approach)
- Requires more historical data for improved accuracy

## Conclusions and future work

The GPR model successfully demonstrates that satellite thermal monitoring can predict pig iron production with actionable accuracy. The cloud-aware processing methodology and uncertainty quantification make it suitable for operational deployment, whilst the honest performance assessment provides realistic expectations.

### Short-term improvements
1. **Data collection**: Acquiring additional historical data remains the most critical need
2. **Improved validation methodology**: Implement proper time series cross-validation (e.g. forward chaining CV for evaluation with no separate test set evaluation and LOO-CV evaluation)
3. **Enhanced features**: Time-lagged thermal variables, seasonal indicators, cumulative thermal activity, key operational metrics (e.g. blast furnace to coke oven ratio)
4. **External data integration**: Maintenance schedules, economic indicators, energy prices, weather data
5. **Codebase improvements**: Implement config management (config.json), robust error handling, comprehensive testing, and experiment tracking for systematic hyperparameter tuning and kernel exploration
6. **Advanced cloud processing**: Develop sophisticated cloud masking with interpolation for missing values, partial clear pixel weighting, and configurable cloud thresholds for improved feature quality

### Long-term research directions
1. **Multi-sensor fusion**: Combine Sentinel-2 with Sentinel-1, Landsat, and MODIS observations
2. **Deep learning exploration**: Time series transformers with proper regularisation (requires substantially more data)
3. **Transfer learning**: Apply methodology to other steel mills and industrial facilities
4. **Operational deployment**: Real-time system with automated updates for 1-2 week production forecasts

### Implementation recommendations
The model is most valuable for anomaly detection rather than precise prediction. Deploy for operational use with regular model updates to prevent data drift, and maintain realistic expectations about predictive accuracy given the inherent constraints of the limited dataset.

## Technical documentation

Complete implementation details, code structure, and usage instructions are available in the project repository. The modular design allows for easy extension and adaptation to other industrial monitoring applications.