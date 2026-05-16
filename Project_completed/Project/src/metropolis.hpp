#pragma once

#include "config.hpp"
#include "rng.hpp"
#include "spin_lattice.hpp"

class Metropolis {
public:
    Metropolis(SpinLattice& lattice, const Config& config, RNG& rng);

    bool single_spin_update();
    int sweep();
    void thermalize(int sweeps);

    double local_energy_for_spin(int idx, const Vec3& s) const;
    double delta_energy(int idx, const Vec3& proposed) const;

    long long attempted() const { return attempted_; }
    long long accepted() const { return accepted_; }
    double acceptance_rate() const;
    void reset_counters();

private:
    SpinLattice& lattice_;
    const Config& config_;
    RNG& rng_;
    long long attempted_ = 0;
    long long accepted_ = 0;
};
