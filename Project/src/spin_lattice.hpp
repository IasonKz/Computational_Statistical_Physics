#pragma once

#include "rng.hpp"
#include <array>
#include <string>
#include <vector>

using Vec3 = std::array<double, 3>;

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline Vec3 mul(double s, const Vec3& a) {
    return {s * a[0], s * a[1], s * a[2]};
}

Vec3 normalize(const Vec3& v);
Vec3 rotate_rodrigues(const Vec3& v, const Vec3& axis, double angle);

class SpinLattice {
public:
    explicit SpinLattice(int L);

    int L() const { return L_; }
    int N() const { return N_; }

    int index(int x, int y, int z) const;
    void coords(int idx, int& x, int& y, int& z) const;
    int neighbor_index(int idx, int dir, int sign) const;

    const Vec3& spin(int idx) const { return spins_[idx]; }
    Vec3& spin(int idx) { return spins_[idx]; }
    void set_spin(int idx, const Vec3& s) { spins_[idx] = s; }

    void initialize(const std::string& mode, double J, double D, RNG& rng);
    void initialize_random(RNG& rng);
    void initialize_ferro();
    void initialize_helical_z(double J, double D);

    const std::vector<Vec3>& spins() const { return spins_; }

private:
    int L_;
    int N_;
    std::vector<Vec3> spins_;
};
