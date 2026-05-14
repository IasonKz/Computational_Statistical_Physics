#pragma once

#include "spin_lattice.hpp"
#include "config.hpp"
#include <array>
#include <vector>

struct MagnetizationResult {
    Vec3 mvec_density{0.0, 0.0, 0.0};
    double m_abs_density = 0.0;
    double m2_density = 0.0;
};

struct StructureFactorResult {
    double max_nonzero = 0.0;
    int axis = 0;
    int n = 1;
    double q = 0.0;
};

struct RunningStats {
    double mean = 0.0;
    double variance = 0.0;
    double stderr = 0.0;
};

double total_energy(const SpinLattice& lattice, const Config& config);
MagnetizationResult magnetization(const SpinLattice& lattice);
StructureFactorResult helical_structure_factor(const SpinLattice& lattice);

RunningStats stats_from_samples(const std::vector<double>& x);
