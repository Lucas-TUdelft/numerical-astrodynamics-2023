###########################################################################
#
# # Numerical Astrodynamics 2022/2023
#
# # Assignment 1 - Propagation Settings
#
###########################################################################


''' 
Copyright (c) 2010-2020, Delft University of Technology
All rights reserved

This file is part of Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

import os

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

# Retrieve current directory
current_directory = os.getcwd()

# Change between case i and ii here:
# note: run question1.py before this script to obtain the necessary data file
case = 1.0

# # student number: 1244779 --> 1244ABC
# student number is: 5009235
A = 2
B = 3
C = 5

simulation_start_epoch = 35.4 * constants.JULIAN_YEAR + A * 7.0 * constants.JULIAN_DAY + B * constants.JULIAN_DAY + C * constants.JULIAN_DAY / 24.0
simulation_end_epoch = simulation_start_epoch + 344.0 * constants.JULIAN_DAY / 24.0

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel( current_directory + "/juice_mat_crema_5_1_150lb_v01.bsp" );

# Create settings for celestial bodies
bodies_to_create = ['Ganymede', 'Jupiter']
global_frame_origin = 'Ganymede'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

density_scale_height = 40.0 * 10**3
density_at_zero_altitude = 2 * 10**(-9)
body_settings.get('Ganymede').atmosphere_settings = environment_setup.atmosphere.exponential(density_scale_height, density_at_zero_altitude)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )

bodies.get("JUICE").mass = 2000.0

# Aero
reference_area = 100.0
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0, 0]
)
environment_setup.add_aerodynamic_coefficient_interface(
    bodies, "JUICE", aero_coefficient_settings)

###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.

if case == 1.0:
    acceleration_settings_on_vehicle = dict(
        Ganymede =
        [
            propagation_setup.acceleration.point_mass_gravity( ),
        ],
        Jupiter =
        [
            propagation_setup.acceleration.spherical_harmonic_gravity(4, 0),
        ]
    )
elif case == 2.0:
    acceleration_settings_on_vehicle = dict(
        Ganymede =
        [
            propagation_setup.acceleration.point_mass_gravity( ),
            propagation_setup.acceleration.aerodynamic()
        ]
    )
# Create global accelerations dictionary.
acceleration_settings = {'JUICE': acceleration_settings_on_vehicle}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='JUICE',
    observer_body_name='Ganymede',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = simulation_start_epoch )

# Define required outputs
if case == 1.0:
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.spherical_harmonic_gravity_type, 'JUICE', 'Jupiter'),
    ]
elif case == 2.0:
    dependent_variables_to_save = [
        propagation_setup.dependent_variable.single_acceleration_norm(
            propagation_setup.acceleration.aerodynamic_type, 'JUICE', 'Ganymede'),
    ]

# Create numerical integrator settings.
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    fixed_step_size
)

# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables = dependent_variables_to_save
)

propagator_settings.print_settings.print_initial_and_final_conditions = True


###########################################################################
# PROPAGATE ORBIT #########################################################
###########################################################################

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings )

# Retrieve all data produced by simulation
propagation_results = dynamics_simulator.propagation_results

# Extract numerical solution for states and dependent variables
state_history = propagation_results.state_history
dependent_variables = propagation_results.dependent_variable_history

###########################################################################
# SAVE RESULTS ############################################################
###########################################################################
if case == 1.0:
    save2txt(solution=state_history,
             filename='JUICEPropagationHistory_Q3.1.dat',
             directory='./'
             )

    save2txt(solution=dependent_variables,
             filename='JUICEPropagationHistory_DependentVariables_Q3.1.dat',
             directory='./'
             )
elif case == 2.0:
    save2txt(solution=state_history,
             filename='JUICEPropagationHistory_Q3.2.dat',
             directory='./'
             )

    save2txt(solution=dependent_variables,
             filename='JUICEPropagationHistory_DependentVariables_Q3.2.dat',
             directory='./'
             )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
dep_var = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

# get unperturbed Cartesian position
with open('C:\\Users\\lucas\\PycharmProjects\\numerical-astrodynamics-2023\\Assignment1\\JUICEPropagationHistory_Q1.dat') as f1:
    content1 = f1.readlines()
    r_mag1 = []
    for line in content1:
        parameters1 = line.split()
        x = float(parameters1[1])
        y = float(parameters1[2])
        z = float(parameters1[3])
        r_mag = np.sqrt(((x**2) + (y**2) + (z**2)))
        r_mag1.append(r_mag)

f1.close()

# get perturbed Cartesian position
if case == 1.0:
    with open('C:\\Users\\lucas\\PycharmProjects\\numerical-astrodynamics-2023\\Assignment1\\JUICEPropagationHistory_Q3.1.dat') as f2:
        content2 = f2.readlines()
        r_mag2 = []
        for line in content2:
            parameters2 = line.split()
            x = float(parameters2[1])
            y = float(parameters2[2])
            z = float(parameters2[3])
            r_mag = np.asarray([x,y,z])
            r_mag2.append(r_mag)

    f2.close()

elif case == 2.0:
    with open('C:\\Users\\lucas\\PycharmProjects\\numerical-astrodynamics-2023\\Assignment1\\JUICEPropagationHistory_Q3.2.dat') as f2:
        content2 = f2.readlines()
        r_mag2 = []
        for line in content2:
            parameters2 = line.split()
            x = float(parameters2[1])
            y = float(parameters2[2])
            z = float(parameters2[3])
            r_mag = np.asarray([x,y,z])
            r_mag2.append(r_mag)

    f2.close()

delta_r = []
for i in range(len(r_mag1)):
    r_diff = r_mag1[i] - r_mag2[i]
    r_diff_mag = np.sqrt(((r_diff[0])**2) + ((r_diff[1])**2) + ((r_diff[2])**2))
    delta_r.append(r_diff_mag)

acc = []
for i in range(len(dep_var)):
    acc.append(dep_var[i][0])

a_int = sp.integrate.cumulative_trapezoid(acc)
epsilon = []
for i in range(len(delta_r) - 1):
    e_i = delta_r[i + 1] / a_int[i]
    epsilon.append(e_i)


plt.plot(time_days[1:],epsilon)
plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration Effectiveness [-]')
plt.yscale('log')
plt.show()



