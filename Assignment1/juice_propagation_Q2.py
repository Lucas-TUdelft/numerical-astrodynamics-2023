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
bodies_to_create = ['Ganymede', 'Jupiter', 'Sun', 'Saturn', 'Europa', 'Io', 'Callisto']
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

# Solar pressure
reference_area_radiation = 100.0
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Ganymede"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(
    bodies, "JUICE", radiation_pressure_settings)

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
        propagation_setup.acceleration.spherical_harmonic_gravity(2, 2),
        propagation_setup.acceleration.aerodynamic()
    ],
    Jupiter =
    [
        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0),
    ],
    Sun =
    [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.cannonball_radiation_pressure()
    ],
    Saturn =
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Europa =
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Io =
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Callisto =
    [
        propagation_setup.acceleration.point_mass_gravity()
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
    propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, 'JUICE', 'Ganymede'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, 'JUICE', 'Jupiter'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, 'JUICE', 'Sun'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, 'JUICE', 'Saturn'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, 'JUICE', 'Europa'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, 'JUICE', 'Io'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, 'JUICE', 'Callisto'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.cannonball_radiation_pressure_type, 'JUICE', 'Sun'),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.aerodynamic_type, 'JUICE', 'Ganymede'),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Ganymede',[(0,0)]),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Ganymede',[(2,0)]),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Ganymede',[(2,2)]),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Jupiter',[(0,0)]),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Jupiter',[(2,0)]),
    propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Jupiter',[(4,0)])
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
         filename='JUICEPropagationHistory_Q2.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q2.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
dep_var = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Change in Kepler elements over the course of the propagation.')



# Semi-major Axis
semi_major_axis = dep_var[:,0] / 1e3
semi_major_axis_diff = semi_major_axis
ax1.plot(time_days, semi_major_axis_diff)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = dep_var[:,1]
eccentricity_diff = eccentricity
ax2.plot(time_days, eccentricity_diff)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(dep_var[:,2])
inclination_diff = inclination
ax3.plot(time_days, inclination_diff)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(dep_var[:,3])
argument_of_periapsis_diff = argument_of_periapsis
ax4.plot(time_days, argument_of_periapsis_diff)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(dep_var[:,4])
raan_diff = raan
ax5.plot(time_days, raan_diff)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(dep_var[:,5])
ax6.scatter(time_days, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [days]')
    ax.set_xlim([min(time_days), max(time_days)])
    ax.grid()
plt.tight_layout()
plt.show()

plt.figure()
# Spherical Harmonic Gravity Acceleration Ganymede
acceleration_norm_sh_G = dep_var[:,6]
plt.plot(time_days, acceleration_norm_sh_G, label='SH Ganymede')

# Spherical Harmonic Gravity Acceleration Jupiter
acceleration_norm_sh_J = dep_var[:,7]
plt.plot(time_days, acceleration_norm_sh_J, label='SH Jupiter')

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_Sun = dep_var[:,8]
plt.plot(time_days, acceleration_norm_pm_Sun, label='PM Sun')

# Point Mass Gravity Acceleration Saturn
acceleration_norm_pm_Sat = dep_var[:,9]
plt.plot(time_days, acceleration_norm_pm_Sat, label='PM Saturn')

# Point Mass Gravity Acceleration Europa
acceleration_norm_pm_E = dep_var[:,10]
plt.plot(time_days, acceleration_norm_pm_E, label='PM Europa')

# Point Mass Gravity Acceleration Io
acceleration_norm_pm_I = dep_var[:,11]
plt.plot(time_days, acceleration_norm_pm_I, label='PM Io')

# Point Mass Gravity Acceleration Callisto
acceleration_norm_pm_C = dep_var[:,12]
plt.plot(time_days, acceleration_norm_pm_C, label='PM Callisto')

# Cannonball Radiation Pressure Acceleration Sun
acceleration_norm_rp_Sun = dep_var[:,13]
plt.plot(time_days, acceleration_norm_rp_Sun, label='Radiation Pressure Sun')

# Aerodynamic Acceleration Earth
acceleration_norm_aero_G = dep_var[:,14]
plt.plot(time_days, acceleration_norm_aero_G, label='Aerodynamic Ganymede')

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend()
plt.yscale('log')

plt.show()

acceleration_norm_G00 = dep_var[:,15]
plt.plot(time_days, acceleration_norm_G00, label='G 0,0')

acceleration_norm_G20 = dep_var[:,16]
plt.plot(time_days, acceleration_norm_G20, label='G 2,0')

acceleration_norm_G22 = dep_var[:,17]
plt.plot(time_days, acceleration_norm_G22, label='G 2,2')

acceleration_norm_J00 = dep_var[:,18]
plt.plot(time_days, acceleration_norm_J00, label='J 0,0')

acceleration_norm_J20 = dep_var[:,19]
plt.plot(time_days, acceleration_norm_J20, label='J 2,0')

acceleration_norm_J40 = dep_var[:,20]
plt.plot(time_days, acceleration_norm_J40, label='J 4,0')

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend()
plt.yscale('log')

plt.show()

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_Sun = dep_var[:,8]
plt.plot(time_days, acceleration_norm_pm_Sun, label='PM Sun')

# Point Mass Gravity Acceleration Io
acceleration_norm_pm_I = dep_var[:,11]
plt.plot(time_days, acceleration_norm_pm_I, label='PM Io')

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend()
plt.yscale('log')

plt.show()
