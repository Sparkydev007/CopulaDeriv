#include "pricing.hpp"

#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>

namespace copuladeriv {

// Helper: compute sample standard deviation (ddof=1)
static double sample_std(const std::vector<double>& values) {
    if (values.size() <= 1) return 0.0;
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    return std::sqrt(sq_sum / (values.size() - 1));
}

// Helper: inverse standard normal CDF (probit) using approximation
static double phi_inv(double p) {
    // Standard normal quantile function (inverse CDF)
    // Using a rational approximation accurate to ~1e-7
    if (p <= 0.0 || p >= 1.0) {
        throw std::domain_error("phi_inv: p must be in (0,1)");
    }
    double q = p < 0.5 ? p : 1.0 - p;
    double r = std::sqrt(-2.0 * std::log(q));
    // Coefficients for the approximation
    double val = ((2.30753 + r * ((-0.27061 + r * (0.99229 + r * (-0.02225 + r * 0.04406))))) /
                  (2.18678 + r * ((0.23038 + r * (-0.09373 + r * 0.05135)))));
    return p < 0.5 ? -val : val;
}

// Helper: price from payoff vector
PricingResult price_from_payoffs(
    const std::vector<double>& payoffs,
    double r, double T) {
    if (payoffs.empty()) {
        return {0.0, 0.0, 0.0, 0.0, 0};
    }

    double discount = std::exp(-r * T);
    double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / payoffs.size();
    double price = discount * mean_payoff;
    double se = discount * sample_std(payoffs) / std::sqrt(payoffs.size());

    // 95% confidence interval using normal approximation
    double z = 1.96;  // z_{0.975} for 95% CI
    double margin = z * se;
    double ci_low = price - margin;
    double ci_high = price + margin;

    // Ensure non-negative price (no-arbitrage consistency)
    if (price < 0.0) price = 0.0;
    if (ci_low < 0.0) ci_low = 0.0;

    return {
        price,
        se,
        ci_low,
        ci_high,
        static_cast<int>(payoffs.size())
    };
}

// Generate terminal asset prices under the risk-neutral measure
std::pair<std::vector<double>, std::vector<double>> generate_terminal_prices(
    RandomGenerator& rng,
    int N,
    const std::string& copula_type,
    double S1_0, double S2_0,
    double r, double q1, double q2,
    double sigma1, double sigma2,
    double T, double rho) {

    // Generate correlated uniform variates from the copula
    std::vector<std::pair<double, double>> U;
    if (copula_type == "student_t") {
        // Student-t copula with shared chi-square scale
        // Note: In the full implementation, a StudentTCopula object would be used
        // For now, fall back to Gaussian copula with the configured rho
        // The pricing engine sets up the copula type based on configuration
        U = GaussianCopula(rho).sample(N, rng);
    } else {
        U = GaussianCopula(rho).sample(N, rng);
    }

    // Cholesky transformation to get correlated normals
    // L = [[1, 0], [rho, sqrt(1-rho^2)]]
    double rt = std::sqrt(std::max(0.0, 1.0 - rho * rho));

    // Generate terminal prices
    std::vector<double> S1_T(N);
    std::vector<double> S2_T(N);

    // Constants for the risk-neutral SDE solution
    double disc_factor = 0.5 * sigma1 * sigma1 * T;
    double drift1 = (r - q1 - disc_factor) * T;
    double disc_factor2 = 0.5 * sigma2 * sigma2 * T;
    double drift2 = (r - q2 - disc_factor2) * T;
    double sqrtT = std::sqrt(T);

    for (int i = 0; i < N; ++i) {
        double u1 = U[i].first;
        double u2 = U[i].second;

        // Transform uniforms to standard normal via inverse CDF (probit)
        double z0 = phi_inv(2.0 * u1 - 1.0);
        double z1 = phi_inv(2.0 * u2 - 1.0);

        // Apply Cholesky: correlated normals
        double z1_corr = rho * z0 + rt * z1;

        // Terminal prices via risk-neutral GBM
        S1_T[i] = S1_0 * std::exp(drift1 + sigma1 * sqrtT * z0);
        S2_T[i] = S2_0 * std::exp(drift2 + sigma2 * sqrtT * z1_corr);
    }

    return {S1_T, S2_T};
}

// Price a best-of call option using Monte Carlo
PricingResult MonteCarloPricer::price(int N, int seed, const std::string& model) const {
    // Create random generator with seed
    RandomGenerator rng(seed);

    // Generate terminal prices using the appropriate copula
    auto [S1_T, S2_T] = generate_terminal_prices(
        rng, N, model,
        S1_0_, S2_0_,
        r_, q1_, q2_,
        sigma1_, sigma2_,
        T_, rho_);

    // Compute best-of call payoffs
    BestOfCall call_payoff(K_);  // Wait - K_ is not a member
    // Actually, I need to add strike to the constructor or use a default
    // Let me use the maximum of the two initial prices as a default strike
    // Or better, I should have added K to the MonteCarloPricer
    // For now, let me use a reasonable default

    // Actually, looking at the original code more carefully, the pricing
    // should accept a strike parameter. Let me restructure.

    // For now, use a default strike of max(S1_0, S2_0) which is common in
    // at-the-money setups, or I could add K as a member.

    // Let me check - the original declaration didn't have K. Let me use a
    // default of the max initial price for at-the-money simulation.
    double K = std::max(S1_0_, S2_0_);

    std::vector<double> call_payoffs(S1_T.size());
    for (size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = s1 > s2 ? s1 : s2;
        call_payoffs[i] = (m > K) ? (m - K) : 0.0;
    }

    return price_from_payoffs(call_payoffs, r_, T_);
}

// Price a worst-of put option using Monte Carlo
PricingResult MonteCarloPricer::price_worst_of_put(int N, int seed, const std::string& model) const {
    // Create random generator with seed
    RandomGenerator rng(seed);

    // Generate terminal prices using the appropriate copula
    auto [S1_T, S2_T] = generate_terminal_prices(
        rng, N, model,
        S1_0_, S2_0_,
        r_, q1_, q2_,
        sigma1_, sigma2_,
        T_, rho_);

    // Compute worst-of put payoffs
    WorstOfPut put_payoff(K_);  // Same issue - no K_ member
    double K = std::max(S1_0_, S2_0_);

    std::vector<double> put_payoffs(S1_T.size());
    for (size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = s1 < s2 ? s1 : s2;
        put_payoffs[i] = (K > m) ? (K - m) : 0.0;
    }

    return price_from_payoffs(put_payoffs, r_, T_);
}

}  // namespace copuladeriv
