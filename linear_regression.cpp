#include "linear_regression.h"
#include "config.h"

#include <numeric>      // std::inner_product
#include <algorithm>    // std::fill, std::clamp
#include <cmath>        // std::abs
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <iostream>     // std::cout
#include <cassert>      // assert

// --- Private Helper Function Declarations ---

/**
 * @brief Predicts an output value for a single row of features.
 * @param x_row Pointer to the start of the feature row.
 * @param weights The model's current weights.
 * @param bias The model's current bias.
 * @return The predicted value.
 */
static float predict(const int *x_row, const std::vector<float>& weights, float bias);

/**
 * @brief Computes the mean squared error cost, including L2 regularization.
 * @param x The input features.
 * @param y The target values.
 * @param weights The model's current weights.
 * @param bias The model's current bias.
 * @return The computed cost.
 */
static float compute_cost(const std::vector<int>& x, const std::vector<int>& y,
                          const std::vector<float>& weights, float bias);

/**
 * @brief Computes the gradient of the cost function with respect to the weights.
 * @param x The input features.
 * @param y The target values.
 * @param weights The model's current weights.
 * @param bias The model's current bias.
 * @param weights_grad_out Output vector to store the computed weight gradients.
 */
static void weights_gradient(const std::vector<int>& x, const std::vector<int>& y,
                             const std::vector<float>& weights, float bias,
                             std::vector<float>& weights_grad_out);

/**
 * @brief Computes the gradient of the cost function with respect to the bias.
 * @param x The input features.
 * @param y The target values.
 * @param weights The model's current weights.
 * @param bias The model's current bias.
 * @return The computed bias gradient.
 */
static float bias_gradient(const std::vector<int>& x, const std::vector<int>& y,
                           const std::vector<float>& weights, float bias);


// --- Public Function Definition ---

std::vector<float> gradient_descent(const std::vector<int>& x, const std::vector<int>& y, float alpha) {
    assert(x.size() == DATA_POINTS * FEATURE_COUNT && "Input x has incorrect size.");
    assert(y.size() == DATA_POINTS && "Input y has incorrect size.");

    std::vector<float> weights(FEATURE_COUNT);
    float bias = 0.0f;

    // Initialize weights with small random values
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);
    for (auto& weight : weights) {
        weight = distribution(generator);
    }

    float previous_cost = compute_cost(x, y, weights, bias);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Calculate gradients
        std::vector<float> weight_grad(FEATURE_COUNT);
        weights_gradient(x, y, weights, bias, weight_grad);
        float bias_grad = bias_gradient(x, y, weights, bias);

        // Update parameters using gradients
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            weights[j] -= alpha * weight_grad[j];
        }
        bias -= alpha * bias_grad;

        // Check for convergence and log progress
        if (epoch % LOG_INTERVAL == 0) {
            float current_cost = compute_cost(x, y, weights, bias);
            std::cout << "Epoch " << epoch << " | Cost: " << current_cost << '\n';

            // Early stopping if cost change is negligible
            if (std::abs(previous_cost - current_cost) < CONVERGENCE_THRESHOLD) {
                std::cout << "Convergence reached at epoch " << epoch << ".\n";
                break;
            }
            previous_cost = current_cost;
        }
    }

    // Return final parameters (weights and bias)
    std::vector<float> final_params = weights;
    final_params.push_back(bias);
    return final_params;
}


// --- Private Helper Function Implementations ---

static float predict(const int *x_row, const std::vector<float>& weights, float bias) {
    float dot_product = std::inner_product(weights.begin(), weights.end(), x_row, 0.0f);
    return dot_product + bias;
}

static float compute_cost(const std::vector<int>& x, const std::vector<int>& y,
                          const std::vector<float>& weights, float bias) {
    float total_error = 0.0f;
    for (int i = 0; i < DATA_POINTS; ++i) {
        const int* x_row = &IDX(x, i, 0);
        float error = predict(x_row, weights, bias) - y[i];
        total_error += error * error;
    }

    // L2 Regularization term
    float regularization_term = 0.0f;
    for (float w : weights) {
        regularization_term += w * w;
    }

    float mse = total_error / (2.0f * DATA_POINTS);
    float l2_cost = (L2_REGULARIZATION_LAMBDA / 2.0f) * regularization_term;

    return mse + l2_cost;
}

static void weights_gradient(const std::vector<int>& x, const std::vector<int>& y,
                             const std::vector<float>& weights, float bias,
                             std::vector<float>& weights_grad_out) {
    std::fill(weights_grad_out.begin(), weights_grad_out.end(), 0.0f);

    for (int i = 0; i < DATA_POINTS; ++i) {
        const int* x_row = &IDX(x, i, 0);
        float error = predict(x_row, weights, bias) - y[i];
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            weights_grad_out[j] += error * IDX(x, i, j);
        }
    }

    for (int j = 0; j < FEATURE_COUNT; ++j) {
        // Average the gradient and add the regularization term
        float reg_grad = L2_REGULARIZATION_LAMBDA * weights[j];
        weights_grad_out[j] = (weights_grad_out[j] / DATA_POINTS) + reg_grad;

        // Clip gradients to prevent them from becoming too large
        weights_grad_out[j] = std::clamp(weights_grad_out[j], -GRADIENT_CLIP_LIMIT, GRADIENT_CLIP_LIMIT);
    }
}

static float bias_gradient(const std::vector<int>& x, const std::vector<int>& y,
                           const std::vector<float>& weights, float bias) {
    float total_error = 0.0f;
    for (int i = 0; i < DATA_POINTS; ++i) {
        const int* x_row = &IDX(x, i, 0);
        total_error += (predict(x_row, weights, bias) - y[i]);
    }
    return total_error / DATA_POINTS;
}
