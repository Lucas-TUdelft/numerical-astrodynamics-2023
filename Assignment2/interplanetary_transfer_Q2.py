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
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

"""
case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t such that we start/end on the sphere of influence

"""
# List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
#cases = ['case_i', 'case_ii', 'case_iii']
cases = ['case_ii']
# Run propagation for each of cases i-iii
for case in cases:

    if case == 'case_i':
        buffer_dep = 0.0
        buffer_arr = 0.0
    if case == 'case_ii':
        buffer_dep = 3600
        buffer_arr = 3600
    if case == 'case_iii':
        buffer_dep = 71 * 3600 # spacecraft is in sphere of influence at index 70, so 71 time steps after departure it leaves
        buffer_arr = 62 * 3600 # spacecraft in in sphere of influence at index 4866, so 62 time steps before the end it is about to enter
    # Define the initial and final propagation time for the current case
    departure_epoch_with_buffer = departure_epoch + buffer_dep
    arrival_epoch_with_buffer = departure_epoch + time_of_flight - buffer_arr

    # Perform propagation
    dynamics_simulator = propagate_trajectory( departure_epoch_with_buffer, arrival_epoch_with_buffer, bodies, lambert_arc_ephemeris,
                     use_perturbations = True)
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)), output_directory)

    state_history = dynamics_simulator.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    delta_r = []
    delta_v = []
    delta_a = []
    times = []

    for key in state_history:
        x_lb = lambert_history[key][0]
        y_lb = lambert_history[key][1]
        z_lb = lambert_history[key][2]
        r_lb = np.asarray([x_lb,y_lb,z_lb])
        r_lb_mag = np.sqrt(((x_lb)**2) + ((y_lb)**2) + ((z_lb)**2))

        vx_lb = lambert_history[key][3]
        vy_lb = lambert_history[key][4]
        vz_lb = lambert_history[key][5]
        v_vec_lb = np.asarray([vx_lb,vy_lb,vz_lb])

        mu_Sun = 1.32712440042 * 10**20
        a_lb = - (mu_Sun * 1000/((r_lb_mag)**3)) * r_lb

        x_nm = state_history[key][0]
        y_nm = state_history[key][1]
        z_nm = state_history[key][2]
        r_nm = np.asarray([x_nm,y_nm,z_nm])

        vx_nm = state_history[key][3]
        vy_nm = state_history[key][4]
        vz_nm = state_history[key][5]
        v_vec_nm = np.asarray([vx_nm,vy_nm,vz_nm])

        ax_nm = dynamics_simulator.dependent_variable_history[key][0]
        ay_nm = dynamics_simulator.dependent_variable_history[key][1]
        az_nm = dynamics_simulator.dependent_variable_history[key][2]
        a_nm = np.asarray([ax_nm,ay_nm,az_nm])

        delta_r_i_vec = r_nm - r_lb
        delta_r_i = np.sqrt(((delta_r_i_vec[0])**2) + ((delta_r_i_vec[1])**2) + ((delta_r_i_vec[2])**2))
        delta_r.append(delta_r_i)

        delta_v_i_vec = v_vec_nm - v_vec_lb
        delta_v_i = np.sqrt(((delta_v_i_vec[0]) ** 2) + ((delta_v_i_vec[1]) ** 2) + ((delta_v_i_vec[2]) ** 2))
        delta_v.append(delta_v_i)

        delta_a_i_vec = a_nm - a_lb
        delta_a_i = np.sqrt(((delta_a_i_vec[0]) ** 2) + ((delta_a_i_vec[1]) ** 2) + ((delta_a_i_vec[2]) ** 2))
        delta_a.append(delta_a_i)

        time = key - departure_epoch_with_buffer
        times.append(time)

    plt.plot(times,delta_r)
    plt.xlabel('time since departure (s)')
    plt.ylabel('delta r (m)')
    plt.show()

    plt.plot(times,delta_v)
    plt.xlabel('time since departure (s)')
    plt.ylabel('delta v (m/s)')
    plt.show()

    plt.plot(times,delta_a)
    plt.xlabel('time since departure (s)')
    plt.ylabel('delta a (m/s^2)')
    plt.show()

    # For case ii, run propagation forward and backward from mid-point
    if case == 'case_ii':
        cutoff = 2463
        cutoff_time = 2463 * 3600 + departure_epoch_with_buffer
        dynamics_simulator1 = propagate_trajectory(cutoff_time, departure_epoch_with_buffer, bodies,
                                                   lambert_arc_ephemeris, use_perturbations=True)

        delta_r2 = []
        delta_v2 = []
        delta_a2 = []
        times2 = []

        for key in state_history:
            x_nm2 = state_history[key][0]
            y_nm2 = state_history[key][1]
            z_nm2 = state_history[key][2]
            r_nm2 = np.asarray([x_nm2, y_nm2, z_nm2])

            delta_r2_i_vec = r_nm2 - r_lb
            delta_r2_i = np.sqrt(((delta_r2_i_vec[0]) ** 2) + ((delta_r2_i_vec[1]) ** 2) + ((delta_r2_i_vec[2]) ** 2))
            delta_r2.append(delta_r2_i)

            time2 = key - departure_epoch_with_buffer
            times2.append(time2)

        dynamics_simulator2 = propagate_trajectory(cutoff_time + fixed_step_size, arrival_epoch_with_buffer, bodies,
                                                   lambert_arc_ephemeris, use_perturbations=True)

        for key in state_history:
            x_nm2 = state_history[key][0]
            y_nm2 = state_history[key][1]
            z_nm2 = state_history[key][2]
            r_nm2 = np.asarray([x_nm2, y_nm2, z_nm2])

            delta_r2_i_vec = r_nm2 - r_lb
            delta_r2_i = np.sqrt(((delta_r2_i_vec[0]) ** 2) + ((delta_r2_i_vec[1]) ** 2) + ((delta_r2_i_vec[2]) ** 2))
            delta_r2.append(delta_r2_i)

            time2 = key - departure_epoch_with_buffer
            times2.append(time2)

        plt.plot(times2, delta_r2)
        plt.xlabel('time since departure (s)')
        plt.ylabel('delta r (m)')
        plt.show()