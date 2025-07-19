#pragma once

#include <vector>

/**
 * @brief Performs gradient descent to find the optimal weights and bias.
 * * @param x A vector of input feature data.
 * @param y A vector of corresponding output data.
 * @param alpha The learning rate for the algorithm.
 * @return A std::vector<float> containing the learned weights, with the bias as the last element.
 */
std::vector<float> gradient_descent(const std::vector<int>& x, const std::vector<int>& y, float alpha);