#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

struct Config {
    int L = 10;
    double T = 1.0;
    double J = 1.0;
    double D = 0.7;
    std::array<double, 3> B{0.0, 0.0, 0.0};
    std::uint64_t seed = 1;
    int thermalization_sweeps = 200;
    int measurement_samples = 500;
    int sweeps_between_measurements = 2;
    double proposal_width = 0.35;
    std::string init_mode = "helical_z";
    std::string out_dir = "results/raw";
    bool save_config = false;
    bool save_measurements = false;

    // ML dataset options: save many thermalized configurations during measurement.
    int save_configs_every = 0;        // 0 disables ML config saving
    int max_saved_configs = 0;         // 0 means no cap
    std::string config_out_dir = "results/configs_ml";
    int ml_label = -1;                 // -1 unknown/test, 0 disordered, 1 helical
};

inline bool is_flag(const std::string& s, const std::string& flag) { return s == flag; }

inline Config parse_args(int argc, char** argv) {
    Config c;
    auto need_value = [&](int i, const std::string& flag) {
        if (i + 1 >= argc) throw std::runtime_error("Missing value after " + flag);
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (is_flag(a, "--L")) { need_value(i,a); c.L = std::stoi(argv[++i]); }
        else if (is_flag(a, "--T")) { need_value(i,a); c.T = std::stod(argv[++i]); }
        else if (is_flag(a, "--J")) { need_value(i,a); c.J = std::stod(argv[++i]); }
        else if (is_flag(a, "--D")) { need_value(i,a); c.D = std::stod(argv[++i]); }
        else if (is_flag(a, "--Bx")) { need_value(i,a); c.B[0] = std::stod(argv[++i]); }
        else if (is_flag(a, "--By")) { need_value(i,a); c.B[1] = std::stod(argv[++i]); }
        else if (is_flag(a, "--Bz")) { need_value(i,a); c.B[2] = std::stod(argv[++i]); }
        else if (is_flag(a, "--seed")) { need_value(i,a); c.seed = static_cast<std::uint64_t>(std::stoull(argv[++i])); }
        else if (is_flag(a, "--therm")) { need_value(i,a); c.thermalization_sweeps = std::stoi(argv[++i]); }
        else if (is_flag(a, "--samples")) { need_value(i,a); c.measurement_samples = std::stoi(argv[++i]); }
        else if (is_flag(a, "--skip")) { need_value(i,a); c.sweeps_between_measurements = std::stoi(argv[++i]); }
        else if (is_flag(a, "--proposal")) { need_value(i,a); c.proposal_width = std::stod(argv[++i]); }
        else if (is_flag(a, "--init")) { need_value(i,a); c.init_mode = argv[++i]; }
        else if (is_flag(a, "--out")) { need_value(i,a); c.out_dir = argv[++i]; }
        else if (is_flag(a, "--save-config")) { c.save_config = true; }
        else if (is_flag(a, "--measurements")) { c.save_measurements = true; }
        else if (is_flag(a, "--save-configs-every")) { need_value(i,a); c.save_configs_every = std::stoi(argv[++i]); }
        else if (is_flag(a, "--max-saved-configs")) { need_value(i,a); c.max_saved_configs = std::stoi(argv[++i]); }
        else if (is_flag(a, "--config-out")) { need_value(i,a); c.config_out_dir = argv[++i]; }
        else if (is_flag(a, "--ml-label")) { need_value(i,a); c.ml_label = std::stoi(argv[++i]); }
        else if (is_flag(a, "--help") || is_flag(a, "-h")) {
            std::cout <<
                "Usage: ./heisenberg_dmi [options]\n\n"
                "Options:\n"
                "  --L INT                    lattice size\n"
                "  --T FLOAT                  temperature\n"
                "  --J FLOAT                  exchange coupling\n"
                "  --D FLOAT                  DMI coupling\n"
                "  --Bx FLOAT                 external field x-component\n"
                "  --By FLOAT                 external field y-component\n"
                "  --Bz FLOAT                 external field z-component\n"
                "  --seed INT                 random seed\n"
                "  --therm INT                thermalization sweeps\n"
                "  --samples INT              number of measured samples\n"
                "  --skip INT                 sweeps between measurements\n"
                "  --proposal FLOAT           max rotation angle in radians\n"
                "  --init MODE                random, ferro, helical_z\n"
                "  --out DIR                  output directory for summary files\n"
                "  --save-config              save final configuration as CSV\n"
                "  --measurements             save all measurement samples as CSV\n"
                "\nML dataset options:\n"
                "  --save-configs-every INT   save one binary spin config every INT measured samples\n"
                "  --max-saved-configs INT    maximum ML configs to save; 0 = no cap\n"
                "  --config-out DIR           output directory for ML binary configs and metadata\n"
                "  --ml-label INT             -1 unknown/test, 0 disordered, 1 helical\n";
            std::exit(0);
        } else throw std::runtime_error("Unknown argument: " + a);
    }
    if (c.L <= 1) throw std::runtime_error("L must be > 1");
    if (c.T <= 0.0) throw std::runtime_error("T must be > 0");
    if (c.thermalization_sweeps < 0) throw std::runtime_error("thermalization sweeps must be >= 0");
    if (c.measurement_samples <= 1) throw std::runtime_error("measurement samples must be > 1");
    if (c.sweeps_between_measurements < 1) throw std::runtime_error("skip must be >= 1");
    if (c.proposal_width <= 0.0) throw std::runtime_error("proposal width must be > 0");
    if (c.save_configs_every < 0) throw std::runtime_error("save-configs-every must be >= 0");
    if (c.max_saved_configs < 0) throw std::runtime_error("max-saved-configs must be >= 0");
    if (!(c.ml_label == -1 || c.ml_label == 0 || c.ml_label == 1)) throw std::runtime_error("ml-label must be -1, 0, or 1");
    return c;
}
