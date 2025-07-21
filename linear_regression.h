#pragma once

#include <vector>

/**
 * @brief Performs gradient descent to train a linear regression model.
 *
 * This function takes the input features (x), target values (y), and a learning rate (alpha),
 * and returns the learned model parameters (weights and bias).
 *
 * @param x A flattened 2D vector of input features (size: DATA_POINTS * FEATURE_COUNT).
 * @param y A vector of target values (size: DATA_POINTS).
 * @param alpha The learning rate for gradient descent.
 * @return A vector containing the learned weights followed by the bias term.
 */
std::vector<float> gradient_descent(const std::vector<int>& x, const std::vector<int>& y, float alpha);