#pragma once

// --- Model & Data Parameters ---

// The number of training iterations.
constexpr int EPOCHS = 10000;
// The number of input features for the model.
constexpr int FEATURE_COUNT = 1;
// The total number of data points to generate.
constexpr int DATA_POINTS = 100;

// --- Ground Truth for Data Generation ---

// The true slope for the line y = mx + c.
constexpr int SLOPE = 2;
// The true y-intercept for the line y = mx + c.
constexpr int Y_INTERCEPT = 1;

// --- Utility Macros ---

// Access element of the i-th row and j-th column of a 1-D vector.
#define IDX(arr, i, j) (arr[(i) * FEATURE_COUNT + (j)])