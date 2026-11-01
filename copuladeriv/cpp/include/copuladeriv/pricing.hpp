#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <array>

#include "random.hpp"
#include "copula.hpp"

namespace copuladeriv {

/** \brief Result of a Monte Carlo pricing run. */
struct PricingResult {
    double price;           // Estimated option price
    double standard_error;  // Monte Carlo standard error
    double confidence_low;  // Lower bound of 95% CI
    double confidence_high; // Upper bound of 95% CI
    int num_scenarios;      // Number of simulation paths
};

/** \brief Payoff functions for rainbow options. */

struct BestOfCall {
    double K;  // Strike

    BestOfCall(double K_) : K(K_) {}

    std::vector<double> operator()(
        const std::vector<double>& S1_T,
        const std::vector<double>& S2_T) const {
        std::vector<double> payoffs(S1_T.size());
        for (size_t i = 0; i < S1_T.size(); ++i) {
            double s1 = S1_T[i];
            double s2 = S2_T[i];
            double m = s1 > s2 ? s1 : s2;
            payoffs[i] = (m > K) ? (m - K) : 0.0;
        }
        return payoffs;
    }
};

struct WorstOfPut {
    double K;  // Strike

    WorstOfPut(double K_) : K(K_) {}

    std::vector<double> operator()(
        const std::vector<double>& S1_T,
        const std::vector<double>& S2_T) const {
        std::vector<double> payoffs(S1_T.size());
        for (size_t i = 0; i < S1_T.size(); ++i) {
            double s1 = S1_T[i];
            double s2 = S2_T[i];
            double m = s1 < s2 ? s1 : s2;
            payoffs[i] = (K > m) ? (K - m) : 0.0;
        }
        return payoffs;
    }
};

struct EuropeanCall {
    double K;  // Strike

    EuropeanCall(double K_) : K(K_) {}

    std::vector<double> operator()(const std::vector<double>& S_T) const {
        std::vector<double> payoffs(S_T.size());
        for (size_t i = 0; i < S_T.size(); ++i) {
            payoffs[i] = (S_T[i] > K) ? (S_T[i] - K) : 0.0;
        }
        return payoffs;
    }
};

/** \brief Monte Carlo pricing engine. */

class MonteCarloPricer {
public:
    /** \brief Construct pricer with model parameters. */
    MonteCarloPricer(
        double S1_0, double S2_0,
        double r, double q1, double q2,
        double sigma1, double sigma2,
        double T, double rho = 0.0, double K = 100.0)
        : S1_0_(S1_0), S2_0_(S2_0),
          r_(r), q1_(q1), q2_(q2),
          sigma1_(sigma1), sigma2_(sigma2),
          T_(T), rho_(rho), K_(K) {
        validate();
    }

    void set_rho(double rho) {
        rho_ = rho;
        copula_.set_rho(rho);
        validate();
    }

    void set_nu(double nu) {
        nu_ = nu;
        student_copula_.set_nu(nu);
    }

    PricingResult price(int N, int seed, const std::string& model = "gaussian") const;

    PricingResult price_worst_of_put(int N, int seed, const std::string& model = "gaussian") const;

    double rho() const { return rho_; }

    double nu() const { return nu_; }

    double K() const { return K_; }

private:
    double S1_0_;
    double S2_0_;
    double r_;
    double q1_;
    double q2_;
    double sigma1_;
    double sigma2_;
    double T_;
    double rho_;
    double nu_ = 5.0;
    double K_;  // Strike price

    GaussianCopula copula_;
    StudentTCopula student_copula_;

    void validate() {
        if (S1_0_ <= 0 || S2_0_ <= 0) {
            throw std::domain_error("MonteCarloPricer: initial prices must be positive");
        }
        if (T_ <= 0) {
            throw std::domain_error("MonteCarloPricer: maturity must be positive");
        }
        if (sigma1_ <= 0 || sigma2_ <= 0) {
            throw std::domain_error("MonteCarloPricer: volatilities must be positive");
        }
        if (std::abs(rho_) > 1.0) {
            throw std::domain_error("MonteCarloPricer: rho must be in [-1, 1]");
        }
    }
};

/** \brief Compute discounted payoff statistics. */
PricingResult price_from_payoffs(
    const std::vector<double>& payoffs,
    double r, double T);

/** \brief Generate terminal asset prices under the risk-neutral measure. */
std::pair<std::vector<double>, std::vector<double>> generate_terminal_prices(
    RandomGenerator& rng,
    int N,
    const std::string& copula_type = "gaussian");

}  // namespace copuladeriv
