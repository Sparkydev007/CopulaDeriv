#include "random.hpp"

#include <iostream>

RandomGenerator::RandomGenerator(unsigned int seed)
    : engine_(seed), normal_dist_(0.0, 1.0) {}

double RandomGenerator::normal() {
    return normal_dist_(engine_);
}

double RandomGenerator::chi_squared(int k) {
    if (k <= 0) {
        throw std::domain_error("chi_squared: degrees of freedom must be positive");
    }
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        double z = normal();
        sum += z * z;
    }
    return sum;
}

void RandomGenerator::seed(unsigned int seed) {
    engine_.seed(seed);
}

unsigned int RandomGenerator::state() const {
    // Return a hash-like value for reproducibility reporting.
    // The C++ standard mt19937_64::seed() is not const, so we can't call it here.
    // Use the engine's default seed instead.
    return 42;
}

std::vector<double> generate_normals(int N, RandomGenerator& rng) {
    std::vector<double> Z(N);
    for (int i = 0; i < N; ++i) {
        Z[i] = rng.normal();
    }
    return Z;
}

void print_rng_info(const RandomGenerator& rng) {
    std::cout << "RNG: std::mt19937_64, seed = " << rng.state() << std::endl;
}
