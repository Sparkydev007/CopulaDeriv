#pragma once

#include <random>
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

/**
 * \brief Reproducible random number generator using std::mt19937_64.
 */
class RandomGenerator {
public:
    using result_type = std::mt19937_64::result_type;

    /**
     * \brief Construct with a given seed for reproducibility.
     * \param seed Seed value (default: 42)
     */
    explicit RandomGenerator(unsigned int seed = 42);

    /**
     * \brief Generate a standard normal variate.
     * \return Z ~ N(0,1)
     */
    double normal();

    /**
     * \brief Generate chi-squared variate with k degrees of freedom.
     * \param k Degrees of freedom (k > 0)
     * \return chi-squared(k)
     * 
     * Generated as sum of squares of k independent standard normals.
     */
    double chi_squared(int k);

    /**
     * \brief Set a new seed (for reproducibility mid-stream).
     * \param seed New seed value
     */
    void seed(unsigned int seed);

    /**
     * \brief Get the current seed.
     * \return Current seed value (note: engine seed may differ from constructor seed
     * if seed() was called after construction).
     * 
     * \note The C++ standard mt19937_64::seed() is not const, so this returns
     * the last seed set. For full state reproduction, the constructor seed should
     * be used.
     */
    unsigned int state() const;

private:
    std::mt19937_64 engine_;
    std::normal_distribution<double> normal_dist_;
};

/**
 * \brief Generate N standard normal samples.
 * \param N Number of samples
 * \param rng Random number generator reference
 * \return Vector of N standard normal variates
 */
std::vector<double> generate_normals(int N, RandomGenerator& rng);

/**
 * \brief Print RNG info for benchmark reporting.
 */
void print_rng_info(const RandomGenerator& rng);