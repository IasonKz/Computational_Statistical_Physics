#include "config.hpp"
#include "metropolis.hpp"
#include "observables.hpp"
#include "rng.hpp"
#include "spin_lattice.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::string safe_double(double x, int precision = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << x;
    std::string s = ss.str();
    for (char& c : s) { if (c == '.') c = 'p'; if (c == '-') c = 'm'; }
    return s;
}

std::string run_basename(const Config& c) {
    std::ostringstream ss;
    ss << "L" << c.L << "_T" << safe_double(c.T)
       << "_J" << safe_double(c.J) << "_D" << safe_double(c.D)
       << "_seed" << c.seed;
    return ss.str();
}

bool nonempty_file(const fs::path& p) {
    return fs::exists(p) && fs::is_regular_file(p) && fs::file_size(p) > 0;
}

void save_configuration_csv(const SpinLattice& lattice, const Config& config) {
    fs::create_directories("results/configs");
    const std::string fname = "results/configs/config_" + run_basename(config) + ".csv";
    std::ofstream f(fname);
    f << "x,y,z,Sx,Sy,Sz\n";
    f << std::setprecision(17);
    for (int idx = 0; idx < lattice.N(); ++idx) {
        int x, y, z; lattice.coords(idx, x, y, z);
        const Vec3& s = lattice.spin(idx);
        f << x << "," << y << "," << z << "," << s[0] << "," << s[1] << "," << s[2] << "\n";
    }
    std::cout << "Saved final configuration to " << fname << "\n";
}

void append_ml_metadata(const Config& c, const std::string& filename, int sample, int saved_index,
                        double e_density, double m_abs_density, const StructureFactorResult& sf) {
    fs::create_directories(c.config_out_dir);
    const fs::path meta_path = fs::path(c.config_out_dir) / "metadata.csv";
    const bool write_header = !nonempty_file(meta_path);
    std::ofstream meta(meta_path, std::ios::app);
    if (!meta) throw std::runtime_error("Could not open ML metadata file: " + meta_path.string());
    if (write_header) {
        meta << "filename,L,T,J,D,Bx,By,Bz,seed,sample,saved_index,label,"
             << "E_density,m_abs_density,helical_order,q_peak,axis_peak,n_peak\n";
    }
    meta << std::setprecision(17)
         << filename << "," << c.L << "," << c.T << "," << c.J << "," << c.D << ","
         << c.B[0] << "," << c.B[1] << "," << c.B[2] << "," << c.seed << ","
         << sample << "," << saved_index << "," << c.ml_label << ","
         << e_density << "," << m_abs_density << "," << sf.max_nonzero << ","
         << sf.q << "," << sf.axis << "," << sf.n << "\n";
}

void save_configuration_binary_ml(const SpinLattice& lattice, const Config& c, int sample, int saved_index,
                                  double e_density, double m_abs_density, const StructureFactorResult& sf) {
    fs::create_directories(c.config_out_dir);
    std::ostringstream name;
    name << "config_" << run_basename(c)
         << "_sample" << std::setw(6) << std::setfill('0') << sample
         << "_saved" << std::setw(4) << std::setfill('0') << saved_index << ".bin";
    const std::string filename = name.str();
    const fs::path path = fs::path(c.config_out_dir) / filename;
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Could not write ML configuration: " + path.string());
    for (int idx = 0; idx < lattice.N(); ++idx) {
        const Vec3& s = lattice.spin(idx);
        const float sx = static_cast<float>(s[0]);
        const float sy = static_cast<float>(s[1]);
        const float sz = static_cast<float>(s[2]);
        out.write(reinterpret_cast<const char*>(&sx), sizeof(float));
        out.write(reinterpret_cast<const char*>(&sy), sizeof(float));
        out.write(reinterpret_cast<const char*>(&sz), sizeof(float));
    }
    append_ml_metadata(c, filename, sample, saved_index, e_density, m_abs_density, sf);
}

int main(int argc, char** argv) {
    try {
        Config config = parse_args(argc, argv);
        fs::create_directories(config.out_dir);

        RNG rng(config.seed);
        SpinLattice lattice(config.L);
        lattice.initialize(config.init_mode, config.J, config.D, rng);
        Metropolis mc(lattice, config, rng);

        std::cout << "Starting run: L=" << config.L << " T=" << config.T
                  << " J=" << config.J << " D=" << config.D
                  << " seed=" << config.seed << " init=" << config.init_mode << "\n";
        if (config.save_configs_every > 0) {
            std::cout << "ML config saving enabled: every " << config.save_configs_every
                      << " measured samples, max=" << config.max_saved_configs
                      << ", config_out=" << config.config_out_dir
                      << ", label=" << config.ml_label << "\n";
        }

        mc.thermalize(config.thermalization_sweeps);
        mc.reset_counters();

        std::vector<double> e_samples, mabs_samples, m2_samples, mx_samples, my_samples, mz_samples, h_samples;
        e_samples.reserve(config.measurement_samples);
        mabs_samples.reserve(config.measurement_samples);
        m2_samples.reserve(config.measurement_samples);
        mx_samples.reserve(config.measurement_samples);
        my_samples.reserve(config.measurement_samples);
        mz_samples.reserve(config.measurement_samples);
        h_samples.reserve(config.measurement_samples);

        std::ofstream meas_file;
        if (config.save_measurements) {
            const std::string mfname = config.out_dir + "/measurements_" + run_basename(config) + ".csv";
            meas_file.open(mfname);
            meas_file << "sample,E_density,m_abs_density,mx_density,my_density,mz_density,m2_density,helical_order,q_peak,axis_peak,n_peak\n";
            meas_file << std::setprecision(17);
        }

        int saved_ml_configs = 0;
        for (int sample = 0; sample < config.measurement_samples; ++sample) {
            for (int k = 0; k < config.sweeps_between_measurements; ++k) mc.sweep();

            const double E = total_energy(lattice, config);
            const double e_density = E / static_cast<double>(lattice.N());
            const MagnetizationResult mag = magnetization(lattice);
            const StructureFactorResult sf = helical_structure_factor(lattice);

            e_samples.push_back(e_density);
            mabs_samples.push_back(mag.m_abs_density);
            m2_samples.push_back(mag.m2_density);
            mx_samples.push_back(mag.mvec_density[0]);
            my_samples.push_back(mag.mvec_density[1]);
            mz_samples.push_back(mag.mvec_density[2]);
            h_samples.push_back(sf.max_nonzero);

            if (config.save_measurements) {
                meas_file << sample << "," << e_density << "," << mag.m_abs_density << ","
                          << mag.mvec_density[0] << "," << mag.mvec_density[1] << ","
                          << mag.mvec_density[2] << "," << mag.m2_density << ","
                          << sf.max_nonzero << "," << sf.q << "," << sf.axis << "," << sf.n << "\n";
            }

            const bool save_ml = config.save_configs_every > 0 && (sample % config.save_configs_every == 0);
            const bool below_cap = (config.max_saved_configs == 0) || (saved_ml_configs < config.max_saved_configs);
            if (save_ml && below_cap) {
                save_configuration_binary_ml(lattice, config, sample, saved_ml_configs, e_density, mag.m_abs_density, sf);
                ++saved_ml_configs;
            }
        }

        const RunningStats e_stats = stats_from_samples(e_samples);
        const RunningStats mabs_stats = stats_from_samples(mabs_samples);
        const RunningStats h_stats = stats_from_samples(h_samples);

        double mean_m2 = std::accumulate(m2_samples.begin(), m2_samples.end(), 0.0) / m2_samples.size();
        double mean_mx = std::accumulate(mx_samples.begin(), mx_samples.end(), 0.0) / mx_samples.size();
        double mean_my = std::accumulate(my_samples.begin(), my_samples.end(), 0.0) / my_samples.size();
        double mean_mz = std::accumulate(mz_samples.begin(), mz_samples.end(), 0.0) / mz_samples.size();
        double mean_mvec2 = mean_mx * mean_mx + mean_my * mean_my + mean_mz * mean_mz;

        const double N = static_cast<double>(lattice.N());
        const double chi_abs = N * mabs_stats.variance / config.T;
        const double chi_vec = N * (mean_m2 - mean_mvec2) / config.T;
        const double cv_per_spin = N * e_stats.variance / (config.T * config.T);
        const double helical_susc_like = N * h_stats.variance / config.T;
        const StructureFactorResult sf_final = helical_structure_factor(lattice);

        const std::string summary_name = config.out_dir + "/summary_" + run_basename(config) + ".csv";
        std::ofstream out(summary_name);
        out << "L,T,J,D,Bx,By,Bz,seed,therm_sweeps,samples,skip,proposal_width,"
            << "E_density_mean,E_density_stderr,M_abs_mean,M_abs_stderr,"
            << "chi_abs,chi_vec,Cv_per_spin,helical_order_mean,helical_order_stderr,"
            << "helical_susc_like,acceptance_rate,q_peak_final,axis_peak_final,n_peak_final,ml_configs_saved\n";
        out << std::setprecision(17)
            << config.L << "," << config.T << "," << config.J << "," << config.D << ","
            << config.B[0] << "," << config.B[1] << "," << config.B[2] << ","
            << config.seed << "," << config.thermalization_sweeps << ","
            << config.measurement_samples << "," << config.sweeps_between_measurements << ","
            << config.proposal_width << "," << e_stats.mean << "," << e_stats.stderr << ","
            << mabs_stats.mean << "," << mabs_stats.stderr << "," << chi_abs << ","
            << chi_vec << "," << cv_per_spin << "," << h_stats.mean << ","
            << h_stats.stderr << "," << helical_susc_like << "," << mc.acceptance_rate() << ","
            << sf_final.q << "," << sf_final.axis << "," << sf_final.n << "," << saved_ml_configs << "\n";

        std::cout << "Wrote summary to " << summary_name << "\n";
        std::cout << "Acceptance rate: " << mc.acceptance_rate() << "\n";
        std::cout << "E/N = " << e_stats.mean << " +/- " << e_stats.stderr << "\n";
        std::cout << "|M|/N = " << mabs_stats.mean << " +/- " << mabs_stats.stderr << "\n";
        std::cout << "Helical order = " << h_stats.mean << " +/- " << h_stats.stderr << "\n";
        if (config.save_configs_every > 0) std::cout << "Saved " << saved_ml_configs << " ML configurations to " << config.config_out_dir << "\n";
        if (config.save_config) save_configuration_csv(lattice, config);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
