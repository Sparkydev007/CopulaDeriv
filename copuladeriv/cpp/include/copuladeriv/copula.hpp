#pragma once

#include <random>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <array>

#include "random.hpp"

/**
 * \brief Gaussian copula sampler.
 * 
 * Generates correlated uniform variates using Cholesky decomposition.
 * 
 * Construction:
 *   1. Generate Z ~ N(0, I_d)  (d independent standard normals per observation)
 *   2. L = Cholesky(correlation matrix)  such that R = L * L^T
 *   3. X = L @ Z  (correlated normals with correlation R)
 *   4. U = Phi(X)  (uniform via standard normal CDF transform)
 * 
 * For the 2D case with correlation rho:
 *   L = [[1, 0], [rho, sqrt(1-rho^2)]]
 */
class GaussianCopula {
public:
    /**
     * \brief Construct with correlation rho.
     * \param rho Correlation parameter in [-1, 1].
     * \throws std::domain_error if |rho| > 1.
     */
    explicit GaussianCopula(double rho = 0.0);

    /**
     * \brief Set the correlation parameter.
     * \param rho Correlation in [-1, 1].
     * \throws std::domain_error if |rho| > 1.
     */
    void set_rho(double rho);

    /**
     * \brief Sample N uniform variates from the Gaussian copula.
     * \param N Number of samples.
     * \param rng Random number generator.
     * \return Vector of N pairs (u1, u2) with Uniform(0,1) variates.
     */
    std::vector<std::pair<double, double>> sample(int N, RandomGenerator& rng);

    /**
     * \brief Get the correlation parameter.
     * \return rho in [-1, 1]
     */
    double rho() const { return rho_; }

private:
    double rho_ = 0.0;
    std::array<std::array<double, 2>, 2> L_;
};

/**
 * \brief Student-t copula sampler.
 * 
 * Generates correlated uniform variates using the scale-mixture-of-Gaussians representation.
 * 
 * Critical implementation detail: A SINGLE chi-squared variate V_i ~ chi-squared(nu)
 * is drawn per observation and used for BOTH assets. This creates the intended dependence.
 * Using independent chi-squared draws per asset would NOT produce the correct multivariate
 * Student-t copula.
 * 
 * Construction per observation i:
 *   1. Draw V_i ~ chi-squared(nu)  (shared scale mixture)
 *   2. Generate Z_i = [z0, z1] i.i.d. ~ N(0,1)
 *   3. Apply Cholesky: X_i = L @ Z_i  (correlated normals)
 *   4. Scale: Y_i = X_i / sqrt(V_i / nu)
 *   5. Transform to uniforms: U_i = Phi(Y_i)
 */
class StudentTCopula {
public:
    /**
     * \brief Construct with correlation rho and degrees of freedom nu.
     * \param rho Correlation parameter in (-1, 1).
     * \param nu Degrees of freedom (nu > 0).
     * \throws std::domain_error if |rho| > 1 or nu <= 0.
     */
    StudentTCopula(double rho = 0.0, double nu = 5.0);

    /**
     * \brief Set the correlation parameter.
     * \param rho Correlation in (-1, 1).
     * \throws std::domain_error if |rho| > 1.
     */
    void set_rho(double rho);

    /**
     * \brief Set the degrees of freedom.
     * \param nu Nu > 0.
     * \throws std::domain_error if nu <= 0.
     */
    void set_nu(double nu);

    /**
     * \brief Sample N uniform variates from the Student-t copula.
     * \param N Number of samples.
     * \param rng Random number generator.
     * \return Vector of N pairs (u1, u2) with Uniform(0,1) variates.
     * 
     * \note Critical: A single chi-squared draw V_i per observation is shared
     * between both assets. This is what creates the multivariate Student-t dependence.
     * Independent chi-squared draws per asset would be incorrect.
     */
    std::vector<std::pair<double, double>> sample(int N, RandomGenerator& rng);

    /**
     * \brief Get the correlation parameter.
     * \return rho in [-1, 1]
     */
    double rho() const { return rho_; }

    /**
     * \brief Get the degrees of freedom.
     * \return nu > 0
     */
    double nu() const { return nu_; }

private:
    double rho_ = 0.0;
    double nu_ = 5.0;
    std::array<std::array<double, 2>, 2> L_;
};

/**
 * \brief Tail dependence coefficients for the Student-t copula.
 * 
 * For the bivariate Student-t copula with correlation rho and degrees of freedom nu:
 * 
 * lambda_L = lambda_U = 2 * Phi(-sqrt((nu+1)*(1-rho)/(1+rho)))
 * 
 * where Phi is the standard normal cumulative distribution function.
 * 
 * This is an approximation to the exact formula which uses the Student-t CDF.
 * The normal approximation captures the correct qualitative behavior:
 * - lambda_L in [0, 1] for all valid rho, nu
 * - lambda_L -> 1 as rho -> 1 (perfect positive tail dependence)
 * - lambda_L -> 0 as rho -> -1 (no lower tail dependence)
 * - lambda_L -> 0 as nu -> inf (converges to Gaussian, zero tail dependence)
 * - lambda_L increases with rho
 * - lambda_L decreases with nu
 * 
 * Exact theoretical formula:
 * lambda_L = 2 * t_cdf_{nu+1}(-sqrt((nu+1)*(1-rho)/(1+rho)))
 * where t_cdf_{df} is the CDF of the Student-t distribution with df degrees of freedom.
 * This simplifies to lambda_L = I_{(1+rho)/2}((nu+1)/2, 1/2) where I is the
 * regularized incomplete beta function.
 * 
 * \param rho Correlation parameter in (-1, 1).
 * \param nu Degrees of freedom (nu > 0).
 * \return Pair (lambda_L, lambda_U) both equal for Student-t copula.
 * \throws std::domain_error if |rho| > 1 or nu <= 0.
 */
std::pair<double, double> student_t_tail_dependence(double rho, double nu);

/**
 * \brief Exit status codes for the copuladeriv CLI.
 */
enum class ExitCode {
    Success = 0,
    UsageError = 1,
    RuntimeError = 2
};