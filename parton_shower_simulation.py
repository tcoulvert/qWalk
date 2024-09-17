import argparse
import datetime
import json
import os
from pathlib import Path

import awkward as ak
import numpy as np

CURRENT_TIME = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Put initial pulse in with some noise (for now, 0) term for other input.
# This gets split into two pulses based on transfer matrix, and theres
#  a relative phase-shift between the two. 
# These pulses (photons) are then recombined into a single long array of
#  pulses at the second optical switch (being careful to place in the
#  correct time-bins and approprately interferring based on relative phases).
# Store the "early" photon in bin "i", and the "late" photons in bin "i+1".
# Maybe use Fock basis for these? although every given pulse will only be in
#  one part of the direct sum b/c there will be only 1 number per pulse (but I
#  guess theres also superposition states across the whole array?).
# Feedback to the initial optical switch and repeat. 
#
#
# YES FOCK -> probelm is N-dimensional hilbert space where N is the number of loops,
#  and after each loop the neighboring states (pulses) interact with each other
#  and you have 1 more pulse after the loop (nth loop (starting index with 1) has n pulses)


def vacuum_noise(size):
    ## TO BE USED LATER FOR MODELING NOISE ##
    return 0 * np.ones(size)

def input_pump(step):
    ## TO BE USED LATER FOR PUMPING EXPERIMENT ##
    if step == 1:  # index step from 1 so it matches with 'number of input pulses'
        return np.ones(1, dtype=float)
    else:
        return vacuum_noise(step)
    
def switch1_scheduler(step):
    ## TO BE USED LATER WHEN ACTUALLY MODELLING PARTON SHOWER ##
    return np.pi/4

def compute_transfer_matrix(theta, phi_R=0, phi_T=0):
    ## This method will become important once we start doing the actual parton shower simulation
    ##  and need to change the reflectivity of the beam-splitters every cycle.

    # maybe need to multiply front by factor like 'np.exp(Im*phi_R(T))'??
    # print(f"theta = {theta}, sin(theta) = {np.sin(theta)}, cos(theta) = {np.cos(theta)}")
    return np.array(
        [[np.sin(theta), np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]]
    )

def optical_switch(in_a, in_b, transfer_matrix):
    in_ = np.array([in_a, in_b])
    out_ = np.einsum('ij,j', transfer_matrix, in_)
    out_c, out_d = out_[0], out_[1]
    # print(f"transfer_matrix: \n{transfer_matrix}")
    # print(f"input: \n{in_}")
    # print(f"output: \n{out_}")
    return out_c, out_d
    
def optical_switch1(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(switch1_scheduler(step))
    return optical_switch(in_a, in_b, transfer_matrix)

def optical_switch2(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(np.pi/4)
    return optical_switch(in_a, in_b, transfer_matrix)

def align_early_late(early_c, late_d):
    return (
        np.concatenate((early_c, vacuum_noise(1))), 
        np.concatenate((vacuum_noise(1), late_d))
    )

def store(pump_array, detector_array, feedback_array, output_dirpath):
    output_dict_for_json = {
        'pump_array': {f'step_{step+1}': pump_array[step].tolist() for step in range(len(pump_array))},
        'feedback_array': {f'step_{step}': feedback_array[step].tolist() for step in range(len(feedback_array))},
        'detector_array': {f'step_{step}': detector_array[step].tolist() for step in range(len(detector_array))},
    }

    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)
    filename = f"{CURRENT_TIME}_QRandWalk"
    with open(os.path.join(output_dirpath, filename), 'w') as f:
        json.dump(output_dict_for_json, f)

def main(n_steps, output_dirpath):
    # Relative phase shift between early and late photons not actually important for this setup!!
    #   -> essentially setting 'delta_phi = 0'
    """
    Need to update optical_switch method to allow for easy numpy array broadcasting, 
    then can delete inner forloops and simplify code.
    """

    pump_array = []
    feedback_array = []
    detector_array = []
    for step in range(1, n_steps+1):
        # print(f"{'='*60}\nstep: {step}\n{'='*60}")
        if step == 1:
            feedback_array.append(vacuum_noise(1))
            detector_array.append(vacuum_noise(1))
        pump_array.append(input_pump(step))

        early_c, late_d = np.zeros(step), np.zeros(step)
        for pulse, (pulse_a, pulse_b) in enumerate(zip(pump_array[-1], feedback_array[-1])):
        #     print(f"pulse_a = {pulse_a}")
        #     print(f"pulse_b = {pulse_b}")
            early_c[pulse], late_d[pulse] = optical_switch1(pulse_a, pulse_b, step)
        # print(f"vacuum_noise(1) = {vacuum_noise(1)}")
        # print(f"early_c = {early_c}")
        # print(f"late_d = {late_d}")
        early_c, late_d = align_early_late(early_c, late_d)
        # print(f"rectified_c = {early_c}")
        # print(f"rectified_d = {late_d}")

        feedback_array.append(np.zeros(step+1))
        detector_array.append(np.zeros(step+1))
        for pulse, (pulse_a, pulse_b) in enumerate(zip(early_c, late_d)):
            feedback_array[-1][pulse], detector_array[-1][pulse] = optical_switch2(pulse_a, pulse_b, step)
        # print(f"pump_array: \n{pump_array}\n{'-'*60}")
        # print(f"feedback_array: \n{feedback_array}\n{'-'*60}")
        # print(f"detector_array: \n{detector_array}\n{'-'*60}")
        # print(f"len(pump_array): \n{len(pump_array)}\n{'-'*60}")
        # print(f"len(feedback_array): \n{len(feedback_array)}\n{'-'*60}")
        # print(f"len(detector_array): \n{len(detector_array)}\n{'-'*60}")

    store(
        pump_array, detector_array, feedback_array,
        output_dirpath
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate the quantum random walk.'
    )
    parser.add_argument('--n-steps', dest='n_steps', action='store', default=15,
        help='The number of steps to iterate the simulation.'
    )
    parser.add_argument('--dump', dest='output_dir_path', action='store', default=f'{str(Path().absolute())}/../output_sim',
        help='Name of the output path in which the processed parquets will be stored.'
    )
    args = parser.parse_args()

    main(args.n_steps, args.output_dir_path)