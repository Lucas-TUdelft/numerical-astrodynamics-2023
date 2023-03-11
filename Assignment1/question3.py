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
from matplotlib import pyplot as plt

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

# Retrieve current directory
current_directory = os.getcwd()

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
bodies_to_create = ['Ganymede']
global_frame_origin = 'Ganymede'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    Ganymede =
    [
        propagation_setup.acceleration.point_mass_gravity( ),
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
dependent_variables_to_save = [
    propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede')
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

save2txt(solution=state_history,
         filename='JUICEPropagationHistory_Q3.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q3.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Change in Kepler elements over the course of the propagation.')

# initial values
a_0 = kepler_elements[0][0] / 1e3
e_0 = kepler_elements[0][1]
i_0 = np.rad2deg(kepler_elements[0][2])
omega_0 = np.rad2deg(kepler_elements[0][3])
raan_0 = np.rad2deg(kepler_elements[0][4])
theta_0 = np.rad2deg(kepler_elements[0][5])

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
semi_major_axis_diff = semi_major_axis - a_0
ax1.plot(time_days, semi_major_axis_diff)
ax1.set_ylabel('Difference in Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
eccentricity_diff = eccentricity - e_0
ax2.plot(time_days, eccentricity_diff)
ax2.set_ylabel('Difference in Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
inclination_diff = inclination - i_0
ax3.plot(time_days, inclination_diff)
ax3.set_ylabel('Difference in Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
argument_of_periapsis_diff = argument_of_periapsis - omega_0
ax4.plot(time_days, argument_of_periapsis_diff)
ax4.set_ylabel('Difference in Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
raan_diff = raan - raan_0
ax5.plot(time_days, raan_diff)
ax5.set_ylabel('Difference in RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(time_days, true_anomaly, s=1)
ax6.set_ylabel('Difference in True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [days]')
    ax.set_xlim([min(time_days), max(time_days)])
    ax.grid()
plt.tight_layout()
plt.show()






