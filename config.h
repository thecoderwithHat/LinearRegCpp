#define CONFIG_H

// --- Data Dimensions ---
constexpr int DATA_POINTS = 100;    // Number of data points in the dataset
constexpr int FEATURE_COUNT = 5;    // Number of features for each data point

// --- Training Hyperparameters ---
constexpr int EPOCHS = 10000;                  // Maximum number of training iterations
constexpr float L2_REGULARIZATION_LAMBDA = 0.01f; // Strength of L2 regularization
constexpr float GRADIENT_CLIP_LIMIT = 1.0f;       // Max value for gradient clipping
constexpr float CONVERGENCE_THRESHOLD = 1e-6f;  // Threshold for early stopping

// --- Logging ---
constexpr int LOG_INTERVAL = 100; // How often to print cost during training
// --- Synthetic Data Generation (for main.cpp) ---
constexpr float SLOPE = 2.5f;       // True slope for synthetic data
constexpr float Y_INTERCEPT = 1.5f; // True y-intercept for synthetic data

// --- Utility Macros ---
// Macro to compute the 1D index for a 2D array stored in a 1D vector
#define IDX(vec, row, col) vec[(row) * FEATURE_COUNT + (col)]

