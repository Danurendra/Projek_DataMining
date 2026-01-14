# LSTM Model Optimization - Final Report

## Executive Summary

Successfully optimized the LSTM model for student dropout prediction, achieving **76.72% accuracy** - a **+2.02% improvement** over the baseline 74.7% accuracy.

## Optimization Strategy

### 1. Architecture Improvements

#### Bidirectional LSTM Layers
- **Before**: Single LSTM layer with 64 units
- **After**: Two bidirectional LSTM layers (128 and 64 units)
- **Benefit**: Bidirectional layers process sequences in both forward and backward directions, capturing more context and improving prediction accuracy

#### Deeper Dense Layers
- **Before**: Single dense layer with 32 units
- **After**: Two dense layers (128 → 64 units)
- **Benefit**: Additional layers allow the model to learn more complex patterns in the data

### 2. Regularization Improvements

#### Progressive Dropout
- **Implementation**: 0.4 (LSTM layers) → 0.3 (first dense) → 0.2 (second dense)
- **Benefit**: Prevents overfitting while maintaining model capacity

### 3. Training Improvements

#### Custom Learning Rate
- **Value**: 0.002 with Adam optimizer
- **Benefit**: Optimal learning rate for stable and efficient training

#### Early Stopping
- **Monitor**: val_accuracy
- **Patience**: 20 epochs
- **Benefit**: Prevents overfitting by stopping when validation accuracy plateaus

#### Learning Rate Scheduling
- **Strategy**: ReduceLROnPlateau
- **Factor**: 0.5 reduction
- **Patience**: 10 epochs
- **Benefit**: Dynamically adjusts learning rate when progress stalls

#### Extended Training
- **Before**: 30 epochs
- **After**: 100 epochs with early stopping
- **Result**: Model stopped at epoch 34 (early stopping triggered)

### 4. Reproducibility
- **Random Seeds**: Set np.random.seed(42) and tf.random.set_seed(42)
- **Benefit**: Ensures consistent results across runs

## Results

### Overall Performance
- **Test Accuracy**: 76.72%
- **Test Loss**: 0.7005
- **Improvement**: +2.02% over baseline (74.7%)

### Per-Class Performance
| Class | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Dropout | 78.17% | 0.80 | 0.78 | 0.79 |
| Graduate | 90.95% | 0.80 | 0.91 | 0.85 |
| Enrolled | 33.96% | 0.53 | 0.34 | 0.41 |

### Confusion Matrix
```
              Predicted
Actual     Dropout  Enrolled  Graduate
Dropout       222      25        37
Enrolled       39      54        66
Graduate       17      23       402
```

## Key Insights

1. **Graduate Class**: Excellent performance (90.95%) - model accurately identifies students likely to graduate
2. **Dropout Class**: Good performance (78.17%) - model reliably predicts dropout risk
3. **Enrolled Class**: Challenging (33.96%) - likely due to class imbalance and inherent uncertainty in this transitional state

## Technical Implementation

### Model Architecture
```python
Model: Sequential
├── Bidirectional LSTM (128 units, return_sequences=True)
├── Dropout (0.4)
├── Bidirectional LSTM (64 units)
├── Dropout (0.4)
├── Dense (128 units, ReLU)
├── Dropout (0.3)
├── Dense (64 units, ReLU)
├── Dropout (0.2)
└── Dense (3 units, Softmax)

Total parameters: 358,275 (1.37 MB)
```

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.002)
- **Loss**: categorical_crossentropy
- **Batch Size**: 32
- **Validation Split**: 20%
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

## Files Delivered

1. **Modelling_LSTM.ipynb** - Updated Jupyter notebook with all optimizations
2. **Modelling_LSTM_Optimized.py** - Standalone Python script
3. **confusion_matrix_optimized.png** - Confusion matrix visualization
4. **training_history_optimized.png** - Training history plots
5. **.gitignore** - Excludes temporary files
6. **OPTIMIZATION_REPORT.md** - This comprehensive report

## Recommendations

### Immediate Next Steps
1. **Re-run the notebook** to generate updated output cells showing the optimized results
2. **Experiment with class weights** to improve "Enrolled" class performance
3. **Try SMOTE or other resampling** techniques to address class imbalance

### Future Enhancements
1. **Hyperparameter Tuning**: Use GridSearch or Bayesian optimization
2. **Feature Engineering**: Add more student behavioral features
3. **Ensemble Methods**: Combine LSTM with other models (e.g., Random Forest, XGBoost)
4. **Attention Mechanism**: Add attention layers to focus on important time steps
5. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation

## Conclusion

The optimization successfully improved the LSTM model accuracy from 74.7% to 76.72%, a meaningful **+2.02% improvement**. The optimized model demonstrates strong performance in predicting student outcomes, particularly for Graduate (90.95%) and Dropout (78.17%) classes. The implementation includes best practices such as bidirectional layers, progressive dropout, learning rate scheduling, and early stopping.

---

**Date**: 2026-01-14  
**Optimization By**: GitHub Copilot  
**Status**: ✅ Complete
