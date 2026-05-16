#include "spin_lattice.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

Vec3 normalize(const Vec3& v) {
    const double n2 = dot(v, v);
    if (n2 <= 0.0) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    const double inv = 1.0 / std::sqrt(n2);
    return {v[0] * inv, v[1] * inv, v[2] * inv};
}

Vec3 rotate_rodrigues(const Vec3& v, const Vec3& axis_in, double angle) {
    const Vec3 axis = normalize(axis_in);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const Vec3 axv = cross(axis, v);
    const double adv = dot(axis, v);

    Vec3 out{
        v[0] * c + axv[0] * s + axis[0] * adv * (1.0 - c),
        v[1] * c + axv[1] * s + axis[1] * adv * (1.0 - c),
        v[2] * c + axv[2] * s + axis[2] * adv * (1.0 - c)
    };
    return normalize(out);
}

SpinLattice::SpinLattice(int L) : L_(L), N_(L * L * L), spins_(N_) {}

int SpinLattice::index(int x, int y, int z) const {
    x = (x % L_ + L_) % L_;
    y = (y % L_ + L_) % L_;
    z = (z % L_ + L_) % L_;
    return x + L_ * (y + L_ * z);
}

void SpinLattice::coords(int idx, int& x, int& y, int& z) const {
    x = idx % L_;
    const int tmp = idx / L_;
    y = tmp % L_;
    z = tmp / L_;
}

int SpinLattice::neighbor_index(int idx, int dir, int sign) const {
    int x, y, z;
    coords(idx, x, y, z);
    if (dir == 0) x += sign;
    else if (dir == 1) y += sign;
    else if (dir == 2) z += sign;
    else throw std::runtime_error("dir must be 0, 1, or 2");
    return index(x, y, z);
}

void SpinLattice::initialize(const std::string& mode, double J, double D, RNG& rng) {
    if (mode == "random") {
        initialize_random(rng);
    } else if (mode == "ferro") {
        initialize_ferro();
    } else if (mode == "helical_z") {
        initialize_helical_z(J, D);
    } else {
        throw std::runtime_error("Unknown init mode: " + mode);
    }
}

void SpinLattice::initialize_random(RNG& rng) {
    for (auto& s : spins_) {
        s = rng.random_unit_vector();
    }
}

void SpinLattice::initialize_ferro() {
    for (auto& s : spins_) {
        s = {1.0, 0.0, 0.0};
    }
}

void SpinLattice::initialize_helical_z(double J, double D) {
    constexpr double pi = 3.141592653589793238462643383279502884;

    const double q0 = std::atan2(D, J);
    int n = static_cast<int>(std::llround(q0 * static_cast<double>(L_) / (2.0 * pi)));
    if (std::abs(q0) > 1e-14) {
        n = std::max(1, n);
    }
    n = std::min(n, L_ / 2);

    const double q = 2.0 * pi * static_cast<double>(n) / static_cast<double>(L_);

    for (int z = 0; z < L_; ++z) {
        const double angle = q * static_cast<double>(z);
        const Vec3 s{std::cos(angle), std::sin(angle), 0.0};
        for (int y = 0; y < L_; ++y) {
            for (int x = 0; x < L_; ++x) {
                spins_[index(x, y, z)] = s;
            }
        }
    }
}
