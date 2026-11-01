#include "copula.hpp"

#include <iostream>
#include <cmath>

GaussianCopula::GaussianCopula(double rho) {
    set_rho(rho);
}

void GaussianCopula::set_rho(double rho) {
    if (std::abs(rho) > 1.0) {
        throw std::domain_error("GaussianCopula: rho must be in [-1, 1]");
    }
    rho_ = rho;
    double rt = std::sqrt(std::max(0.0, 1.0 - rho * rho));
    L_[0][0] = 1.0; L_[0][1] = 0.0;
    L_[1][0] = rho; L_[1][1] = rt;
}

std::vector<std::pair<double, double>> GaussianCopula::sample(int N, RandomGenerator& rng) {
    std::vector<std::pair<double, double>> U(N);

    // Generate N pairs of independent standard normals
    std::vector<double> Z(2 * N);
    for (int i = 0; i < 2 * N; ++i) {
        Z[i] = rng.normal();
    }

    // Apply Cholesky and transform to uniforms
    for (int i = 0; i < N; ++i) {
        int idx = i * 2;
        double z0 = Z[idx];
        double z1 = Z[idx + 1];

        // Cholesky transformation: X = L @ Z
        double x0 = L_[0][0] * z0 + L_[0][1] * z1;  // = z0
        double x1 = L_[1][0] * z0 + L_[1][1] * z1;  // = rho*z0 + sqrt(1-rho^2)*z1

        // Transform to uniforms via standard normal CDF: Phi(x) = 0.5*(1+erf(x/sqrt(2)))
        U[i].first  = 0.5 * (1.0 + std::erf(x0 / std::sqrt(2.0)));
        U[i].second = 0.5 * (1.0 + std::erf(x1 / std::sqrt(2.0)));
    }

    return U;
}

StudentTCopula::StudentTCopula(double rho, double nu) {
    set_rho(rho);
    set_nu(nu);
}

void StudentTCopula::set_rho(double rho) {
    if (std::abs(rho) > 1.0) {
        throw std::domain_error("StudentTCopula: rho must be in [-1, 1]");
    }
    rho_ = rho;
    double rt = std::sqrt(std::max(0.0, 1.0 - rho * rho));
    L_[0][0] = 1.0; L_[0][1] = 0.0;
    L_[1][0] = rho; L_[1][1] = rt;
}

void StudentTCopula::set_nu(double nu) {
    if (nu <= 0.0) {
        throw std::domain_error("StudentTCopula: nu must be positive");
    }
    nu_ = nu;
}

std::vector<std::pair<double, double>> StudentTCopula::sample(int N, RandomGenerator& rng) {
    std::vector<std::pair<double, double>> U(N);

    for (int i = 0; i < N; ++i) {
        // CRITICAL: Draw a SINGLE chi-squared V_i ~ chi-squared(nu) for this observation
        double V = rng.chi_squared(static_cast<int>(nu_));

        // Generate two independent standard normals
        double z0 = rng.normal();
        double z1 = rng.normal();

        // Apply Cholesky to get correlated normals
        double x0 = L_[0][0] * z0 + L_[0][1] * z1;  // = z0
        double x1 = L_[1][0] * z0 + L_[1][1] * z1;  // = rho*z0 + sqrt(1-rho^2)*z1

        // Scale by the shared scale mixture: Y = X / sqrt(V / nu)
        double scale = std::sqrt(V / nu_);
        double y0 = x0 / scale;
        double y1 = x1 / scale;

        // Transform to uniforms via standard normal CDF
        U[i].first  = 0.5 * (1.0 + std::erf(y0 / std::sqrt(2.0)));
        U[i].second = 0.5 * (1.0 + std::erf(y1 / std::sqrt(2.0)));
    }

    return U;
}

// Compute tail dependence coefficients for the Student-t copula
// lambda_L = lambda_U = 2 * Phi(-sqrt((nu+1)*(1-rho)/(1+rho)))
// where Phi is the standard normal CDF
std::pair<double, double> student_t_tail_dependence(double rho, double nu) {
    if (std::abs(rho) > 1.0) {
        throw std::domain_error("student_t_tail_dependence: rho must be in [-1, 1]");
    }
    if (nu <= 0.0) {
        throw std::domain_error("student_t_tail_dependence: nu must be positive");
    }

    // Compute the argument to the normal CDF
    double arg_factor = (nu + 1.0) * (1.0 - rho) / (1.0 + rho);
    if (arg_factor < 0.0) {
        arg_factor = 0.0;
    }
    double arg = -std::sqrt(arg_factor);

    // Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    double phi_arg = 0.5 * (1.0 + std::erf(arg / std::sqrt(2.0)));

    // Tail dependence lambda_L = 2 * Phi(arg)
    // This captures the correct qualitative behavior:
    // - lambda_L -> 1 as rho -> 1 (arg -> 0, Phi(0) = 0.5, lambda -> 1)
    // - lambda_L -> 0 as rho -> -1 (arg -> -inf, Phi(-inf) = 0, lambda -> 0)
    // - lambda_L -> 0 as nu -> inf (arg -> -inf, lambda -> 0)
    // - lambda_L increases with rho
    // - lambda_L decreases with nu
    double lambda = 2.0 * phi_arg;

    // Clamp to [0, 1] for safety
    lambda = std::max(0.0, std::min(1.0, lambda));

    return {lambda, lambda};
}
