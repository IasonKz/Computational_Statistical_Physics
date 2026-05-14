#include "observables.hpp"
#include <complex>
#include <cmath>
#include <stdexcept>

static const std::array<Vec3, 3> ehat{{
    Vec3{1.0, 0.0, 0.0},
    Vec3{0.0, 1.0, 0.0},
    Vec3{0.0, 0.0, 1.0}
}};

double total_energy(const SpinLattice& lattice, const Config& config) {
    double E = 0.0;

    for (int idx = 0; idx < lattice.N(); ++idx) {
        const Vec3& S = lattice.spin(idx);

        for (int dir = 0; dir < 3; ++dir) {
            const int j = lattice.neighbor_index(idx, dir, +1);
            const Vec3& Sj = lattice.spin(j);
            E += -config.J * dot(S, Sj);
            E += -config.D * dot(ehat[dir], cross(S, Sj));
        }

        E += -dot(config.B, S);
    }

    return E;
}

MagnetizationResult magnetization(const SpinLattice& lattice) {
    Vec3 M{0.0, 0.0, 0.0};
    for (int idx = 0; idx < lattice.N(); ++idx) {
        const Vec3& s = lattice.spin(idx);
        M[0] += s[0];
        M[1] += s[1];
        M[2] += s[2];
    }

    const double invN = 1.0 / static_cast<double>(lattice.N());
    Vec3 m{M[0] * invN, M[1] * invN, M[2] * invN};
    const double m2 = dot(m, m);

    MagnetizationResult out;
    out.mvec_density = m;
    out.m_abs_density = std::sqrt(m2);
    out.m2_density = m2;
    return out;
}

StructureFactorResult helical_structure_factor(const SpinLattice& lattice) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    using cd = std::complex<double>;

    StructureFactorResult best;
    const int L = lattice.L();
    const double invN2 = 1.0 / static_cast<double>(lattice.N()) / static_cast<double>(lattice.N());

    for (int axis = 0; axis < 3; ++axis) {
        for (int n = 1; n < L; ++n) {
            const double q = 2.0 * pi * static_cast<double>(n) / static_cast<double>(L);
            cd sumx(0.0, 0.0), sumy(0.0, 0.0), sumz(0.0, 0.0);

            for (int idx = 0; idx < lattice.N(); ++idx) {
                int x, y, z;
                lattice.coords(idx, x, y, z);
                int r = (axis == 0 ? x : (axis == 1 ? y : z));
                const double phase = q * static_cast<double>(r);
                const cd w(std::cos(phase), std::sin(phase));
                const Vec3& s = lattice.spin(idx);
                sumx += s[0] * w;
                sumy += s[1] * w;
                sumz += s[2] * w;
            }

            const double Sq = (std::norm(sumx) + std::norm(sumy) + std::norm(sumz)) * invN2;
            if (Sq > best.max_nonzero) {
                best.max_nonzero = Sq;
                best.axis = axis;
                best.n = n;
                best.q = q;
            }
        }
    }

    return best;
}

RunningStats stats_from_samples(const std::vector<double>& x) {
    if (x.size() < 2) {
        throw std::runtime_error("Need at least two samples for statistics");
    }

    const double n = static_cast<double>(x.size());
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= n;

    double var = 0.0;
    for (double v : x) {
        const double d = v - mean;
        var += d * d;
    }
    var /= (n - 1.0);

    RunningStats out;
    out.mean = mean;
    out.variance = var;
    out.stderr = std::sqrt(var / n);
    return out;
}
