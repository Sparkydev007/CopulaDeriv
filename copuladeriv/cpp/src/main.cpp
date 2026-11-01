#include "random.hpp"
#include "copula.hpp"
#include "pricing.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::cout << "CopulaDeriv C++20 Monte Carlo Engine" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Simple test: create a Gaussian copula and sample
    try {
        RandomGenerator rng(42);
        GaussianCopula gc(0.5);
        
        auto U = gc.sample(1000, rng);
        std::cout << "Gaussian copula sample shape: " << U.size() << " pairs" << std::endl;
        std::cout << "Correlation rho = " << gc.rho() << std::endl;
        
        // Test Student-t copula
        StudentTCopula st(0.5, 6.0);
        auto Ut = st.sample(1000, rng);
        std::cout << "Student-t copula sample shape: " << Ut.size() << " pairs" << std::endl;
        std::cout << "Student-t rho = " << st.rho() << ", nu = " << st.nu() << std::endl;
        
        // Test tail dependence
        auto lam = student_t_tail_dependence(0.5, 6.0);
        std::cout << "Student-t tail dependence rho=0.5, nu=6: lambda_L=" << lam.first
                  << ", lambda_U=" << lam.second << std::endl;
        
        std::cout << std::endl << "All basic tests passed!" << std::endl;
        return static_cast<int>(ExitCode::Success);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return static_cast<int>(ExitCode::RuntimeError);
    }
}
