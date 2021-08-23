# -*- coding: utf-8 -*-

# Configurations for Gradual Patterns:
# INITIALIZATIONS = 3
ALGORITHM = 'aco'  # aco, ga, pso, graank, acolcm, lcm, prs, pls
MIN_SUPPORT = 0.5
CPU_CORES = 4  # Depends on your computer

# DATASET = "../../data/DATASET.csv"
# DATASET = "../../data/hcv_data.csv"

# Uncomment for Main:
DATASET = "../data/DATASET.csv"

# Uncomment for Terminal:
# DATASET = "data/DATASET.csv"

# Global Swarm Configurations
MAX_ITERATIONS = 10
MAX_EVALUATIONS = 100
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAD Configurations:
EVAPORATION_FACTOR = 0.5

# GA-GRAD Configurations:
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAD Configurations:
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAD Configurations
STEP_SIZE = 0.5

# VISUALIZATIONS
SHOW_P_MATRIX = False  # ONLY FOR: aco
SHOW_EVALUATIONS = False  # FOR aco, prs, pls, pso
SHOW_ITERATIONS = False  # FOR aco, prs, pls, pso
