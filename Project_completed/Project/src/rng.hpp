#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <random>

class RNG {
public:
    explicit RNG(std::uint64_t seed) : gen_(seed), uniform01_(0.0, 1.0) {}

    double uniform01() {
        return uniform01_(gen_);
    }

    double uniform(double a, double b) {
        std::uniform_real_distribution<double> dist(a, b);
        return dist(gen_);
    }

    int randint(int n) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(gen_);
    }

    std::array<double, 3> random_unit_vector() {
        constexpr double pi = 3.141592653589793238462643383279502884;
        const double z = uniform(-1.0, 1.0);
        const double phi = uniform(0.0, 2.0 * pi);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return {r * std::cos(phi), r * std::sin(phi), z};
    }

private:
    std::mt19937_64 gen_;
    std::uniform_real_distribution<double> uniform01_;
};
