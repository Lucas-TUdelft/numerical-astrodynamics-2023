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
cases = ['case_i', 'case_ii', 'case_iii']

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
    dynamics_simulator = XXXX
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)), output_directory)

    state_history = dynamics_simulator.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    # For case ii, run propagation forward and backward from mid-point
    if case == 'case_ii':
        XXXX
