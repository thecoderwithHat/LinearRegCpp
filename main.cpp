#include <iostream>
#include <vector>
#include <cstdlib>

#include "config.h"
#include "linear_regression.h"

int main() {
    // 1. Generate the dataset
    std::vector<int> x(FEATURE_COUNT * DATA_POINTS);
    for (int i = 0; i < DATA_POINTS; ++i) {
        for (int j = 0; j < FEATURE_COUNT; ++j) {
            IDX(x, i, j) = i;
        }
    }

    std::vector<int> y(DATA_POINTS);
    for (int i = 0; i < DATA_POINTS; ++i) {
        y[i] = SLOPE * i + Y_INTERCEPT;
    }

    // 2. Set the learning rate
    float alpha = 0.0001f;

    // 3. Run gradient descent to learn the parameters
    std::vector<float> params = gradient_descent(x, y, alpha);
    
    // 4. Print the learned parameters
    std::cout << "Training complete." << std::endl;
    std::cout << "Target -> Slope: " << SLOPE << ", Y-Intercept: " << Y_INTERCEPT << std::endl;
    std::cout << "Result -> ";
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        std::cout << "Weight #" << i << ": " << params[i] << ", ";
    }
    std::cout << "Bias: " << params.back() << std::endl;

    return EXIT_SUCCESS;
}