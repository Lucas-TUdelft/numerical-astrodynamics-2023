''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from interplanetary_transfer_helper_functions import *
import matplotlib.pyplot as plt

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies( )

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

# Create propagation settings and propagate dynamics
dynamics_simulator = propagate_trajectory( departure_epoch, arrival_epoch, bodies, lambert_arc_ephemeris,
                     use_perturbations = False)

# Write results to file
write_propagation_results_to_file(
    dynamics_simulator, lambert_arc_ephemeris, "Q1",output_directory)

# Extract state history from dynamics simulator
state_history = dynamics_simulator.state_history

# Evaluate the Lambert arc model at each of the epochs in the state_history
lambert_history = get_lambert_arc_history( lambert_arc_ephemeris, state_history )

x_states_nm = []
y_states_nm = []
z_states_nm = []

x_states_lb = []
y_states_lb = []
z_states_lb = []

x_diff = []
y_diff = []
z_diff = []

Earth_x = []
Earth_y = []
Earth_z = []

Mars_x = []
Mars_y = []
Mars_z = []

times = []
for key in state_history:
    x_state_nm = state_history[key][0]
    x_states_nm.append(x_state_nm)

    y_state_nm = state_history[key][1]
    y_states_nm.append(y_state_nm)

    z_state_nm = state_history[key][2]
    z_states_nm.append(z_state_nm)

    x_state_lb = lambert_history[key][0]
    x_states_lb.append(x_state_lb)

    y_state_lb = lambert_history[key][1]
    y_states_lb.append(y_state_lb)

    z_state_lb = lambert_history[key][2]
    z_states_lb.append(z_state_lb)

    delta_x = x_state_nm - x_state_lb
    x_diff.append(delta_x)

    delta_y = y_state_nm - y_state_lb
    y_diff.append(delta_y)

    delta_z = z_state_nm - z_state_lb
    z_diff.append(delta_z)

    time = key - departure_epoch
    times.append(time)

    xEarth = dynamics_simulator.dependent_variable_history[key][0]
    Earth_x.append(xEarth)

    yEarth = dynamics_simulator.dependent_variable_history[key][1]
    Earth_y.append(yEarth)

    zEarth = dynamics_simulator.dependent_variable_history[key][2]
    Earth_z.append(zEarth)

    xMars = dynamics_simulator.dependent_variable_history[key][3]
    Mars_x.append(xMars)

    yMars = dynamics_simulator.dependent_variable_history[key][4]
    Mars_y.append(yMars)

    zMars = dynamics_simulator.dependent_variable_history[key][5]
    Mars_z.append(zMars)

fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.set_title(f"Total numerical trajectory in 3D")

ax1.plot(x_states_nm, y_states_nm, z_states_nm, linestyle="-", color="green", label="Trajectory")
ax1.plot(Earth_x,Earth_y,Earth_z, linestyle="-", color="blue", label="Earth")
ax1.plot(Mars_x,Mars_y,Mars_z, linestyle="-", color="red", label="Mars")
ax1.scatter(0.0,0.0,0.0, marker="o", color="yellow", label="Sun")

ax1.legend()
ax1.set_xlim([-2E11, 2E11]), ax1.set_ylim([-2E11, 2E11]), ax1.set_zlim([-2E11, 2E11])
ax1.set_xlabel("x [m], 1e11"), ax1.set_ylabel("y [m], 1e11"), ax1.set_zlabel("z [m], 1e11")
fig1.tight_layout()

plt.show()

plt.plot(times,x_diff, label="Difference in x-position")
plt.plot(times,y_diff, label="Difference in y-position")
plt.plot(times,z_diff, label="difference in z-position")
plt.xlabel("Time (s) since departure")
plt.ylabel("Difference between numerical propagation and Lambert targeter")
plt.legend()
plt.show()
