# Lab3: MATLAB solutions

The root README.md is the original assignment, unchanged.

- Main: open `main/lab3_main.m` in MATLAB and press Run.
- Additional: open `additional/lab3_adaptive.m` and press Run.

Both scripts generate their own data and need only base MATLAB (R2016b or later). No neural-network toolbox is required. They clear the workspace and close earlier figures, so run them separately.

## Main task

Two Gaussian RBF neurons and one linear output are used. The input is exactly `0.1:1/22:1`: 20 values, ending at approximately 0.9636. The assignment's unmatched parentheses are interpreted as dividing the entire sum by 2.

Centers are manually selected as `[0.20; 0.60]`, near the target function's high and low regions. Both widths are 0.15. Output weights start at `[0.1; -0.1]` and bias at zero. Only these three output parameters are learned, using sequential LMS updates for 5000 epochs with learning rate 0.05. Unlike classification, the output stays continuous; do not apply sign.

## Additional task

The same data, initialization, epoch count and output learning rate are used. Gradient descent also updates both centers and widths, with learning rate 0.005. All derivatives are calculated before parameters change. Widths are clipped to [0.03,1] to avoid zero or negative values; centers are clipped to [0,1]. This is projected gradient descent on seven parameters.

## What to compare

Both scripts print training MSE, parameters and dense-grid MSE. Compare their target/prediction plots. The additional script also plots centers and widths across epochs. The dense grid checks the approximation over the same interval; it is not a fully independent test dataset.

## Validation

MATLAB R2026a's standalone static analyzer reported no issues. Full MATLAB startup failed with `File system inconsistency` before script execution. Independent Python/NumPy calculations using the same deterministic initialization gave training MSE about 0.004313 (fixed RBF) and 0.000125 (adaptive RBF). Finite-difference checks confirmed the analytical weight, bias, center and width gradients. These are independent numerical checks, not confirmed MATLAB execution results. Run the scripts in MATLAB and report your own printed results.

Personal explanations, formulas and defense questions are in Notion.
