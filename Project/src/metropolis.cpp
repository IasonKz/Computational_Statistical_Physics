#include "metropolis.hpp"
#include <array>
#include <cmath>

Metropolis::Metropolis(SpinLattice& lattice, const Config& config, RNG& rng)
    : lattice_(lattice), config_(config), rng_(rng) {}

void Metropolis::reset_counters() {
    attempted_ = 0;
    accepted_ = 0;
}

double Metropolis::acceptance_rate() const {
    if (attempted_ == 0) return 0.0;
    return static_cast<double>(accepted_) / static_cast<double>(attempted_);
}

double Metropolis::local_energy_for_spin(int idx, const Vec3& s) const {
    static const std::array<Vec3, 3> ehat{{
        Vec3{1.0, 0.0, 0.0},
        Vec3{0.0, 1.0, 0.0},
        Vec3{0.0, 0.0, 1.0}
    }};

    double E = 0.0;

    for (int dir = 0; dir < 3; ++dir) {
        const int fwd = lattice_.neighbor_index(idx, dir, +1);
        const int bwd = lattice_.neighbor_index(idx, dir, -1);

        const Vec3& Sf = lattice_.spin(fwd);
        const Vec3& Sb = lattice_.spin(bwd);

        // Forward bond: idx -> idx + e_dir
        E += -config_.J * dot(s, Sf);
        E += -config_.D * dot(ehat[dir], cross(s, Sf));

        // Backward bond: idx - e_dir -> idx
        E += -config_.J * dot(Sb, s);
        E += -config_.D * dot(ehat[dir], cross(Sb, s));
    }

    E += -dot(config_.B, s);
    return E;
}

double Metropolis::delta_energy(int idx, const Vec3& proposed) const {
    const Vec3& old = lattice_.spin(idx);
    return local_energy_for_spin(idx, proposed) - local_energy_for_spin(idx, old);
}

bool Metropolis::single_spin_update() {
    const int idx = rng_.randint(lattice_.N());
    const Vec3 old = lattice_.spin(idx);
    const Vec3 axis = rng_.random_unit_vector();
    const double angle = rng_.uniform(-config_.proposal_width, config_.proposal_width);
    const Vec3 proposed = rotate_rodrigues(old, axis, angle);

    const double dE = delta_energy(idx, proposed);
    bool accept = false;
    if (dE <= 0.0) {
        accept = true;
    } else {
        accept = (std::exp(-dE / config_.T) > rng_.uniform01());
    }

    ++attempted_;
    if (accept) {
        lattice_.set_spin(idx, proposed);
        ++accepted_;
    }
    return accept;
}

int Metropolis::sweep() {
    int acc = 0;
    for (int n = 0; n < lattice_.N(); ++n) {
        if (single_spin_update()) ++acc;
    }
    return acc;
}

void Metropolis::thermalize(int sweeps) {
    for (int s = 0; s < sweeps; ++s) {
        sweep();
    }
}
