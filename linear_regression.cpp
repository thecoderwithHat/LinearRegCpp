#include "linear_regression.h"
#include "config.h"

#include <numeric>      // For std::inner_product
#include <algorithm>    // For std::fill

// --- Private Helper Functions (Implementation Details) ---
// These are not exposed in the header file. We can declare them static
// to limit their scope to this file.

/**
 * @brief Makes a prediction for a single input row using a dot product.
 */
static float predict(const int *x_row, const std::vector<float>& weights, float bias) {
    // std::inner_product computes the dot product of the two ranges.
    float dot_product = std::inner_product(weights.begin(), weights.end(), x_row, 0.0f);
    return dot_product + bias;
}

/**
 * @brief Computes the gradient of the cost function with respect to the weights.
 */
static void weights_gradient(const std::vector<int>& x, const std::vector<int>& y, const std::vector<float>& weights, float bias, std::vector<float>& weights_grad_out) {
    std::fill(weights_grad_out.begin(), weights_grad_out.end(), 0.0f);

    for (int i = 0; i < DATA_POINTS; ++i) {
        const int* x_row = &IDX(x, i, 0);
        const float prediction = predict(x_row, weights, bias);
        const float error = prediction - y[i];

        for (int j = 0; j < FEATURE_COUNT; ++j) {
            weights_grad_out[j] += error * IDX(x, i, j);
        }
    }

    for (int i = 0; i < FEATURE_COUNT; ++i) {
        weights_grad_out[i] /= DATA_POINTS;
    }
}

/**
 * @brief Computes the gradient of the cost function with respect to the bias.
 */
static float bias_gradient(const std::vector<int>& x, const std::vector<int>& y, const std::vector<float>& weights, float bias) {
    float bias_grad = 0.0f;
    for (int i = 0; i < DATA_POINTS; ++i) {
        const int* x_row = &IDX(x, i, 0);
        const float prediction = predict(x_row, weights, bias);
        bias_grad += (prediction - y[i]);
    }
    return bias_grad / DATA_POINTS;
}


// --- Public Function Definition ---

std::vector<float> gradient_descent(const std::vector<int>& x, const std::vector<int>& y, float alpha) {
    std::vector<float> weights(FEATURE_COUNT, 0.0f);
    float bias = 0.0f;

    for (int i = 0; i < EPOCHS; ++i) {
        std::vector<float> weight_g(FEATURE_COUNT);
        
        weights_gradient(x, y, weights, bias, weight_g);
        float bias_g = bias_gradient(x, y, weights, bias);

        // Update weights and bias
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            weights[j] -= alpha * weight_g[j];
        }
        bias -= alpha * bias_g;
    }

    // Combine weights and bias into a single vector to return.
    std::vector<float> params = weights;
    params.push_back(bias);
    
    return params;
}