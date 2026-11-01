#include <vector>
#include <algorithm>
#include <cmath>

// Best-of call payoff: H_T = max(max(S1_T, S2_T) - K, 0)
std::vector<double> best_of_call(
    const std::vector<double>& S1_T,
    const std::vector<double>& S2_T,
    double K) {
    std::vector<double> payoffs(S1_T.size());
    for (size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = s1 > s2 ? s1 : s2;  // max(s1, s2)
        payoffs[i] = (m > K) ? (m - K) : 0.0;
    }
    return payoffs;
}

// Worst-of put payoff: H_T = max(K - min(S1_T, S2_T), 0)
std::vector<double> worst_of_put(
    const std::vector<double>& S1_T,
    const std::vector<double>& S2_T,
    double K) {
    std::vector<double> payoffs(S1_T.size());
    for (size_t i = 0; i < S1_T.size(); ++i) {
        double s1 = S1_T[i];
        double s2 = S2_T[i];
        double m = s1 < s2 ? s1 : s2;  // min(s1, s2)
        payoffs[i] = (K > m) ? (K - m) : 0.0;
    }
    return payoffs;
}

// Single-asset European call: H_T = max(S_T - K, 0)
std::vector<double> european_call(
    const std::vector<double>& S_T,
    double K) {
    std::vector<double> payoffs(S_T.size());
    for (size_t i = 0; i < S_T.size(); ++i) {
        payoffs[i] = (S_T[i] > K) ? (S_T[i] - K) : 0.0;
    }
    return payoffs;
}