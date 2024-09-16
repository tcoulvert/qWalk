import argparse
from pathlib import Path

import awkward as ak
import numpy as np


def generate_initial_pulse(n_steps):
    return (1,0)

def generate_noise(n_steps):
    return np.zeros(n_steps)

def compute_transfer_matrix(step, switch_num, theta, phi_R=0, phi_T=0):
    # return np.array(
    #     [[np.exp(Im*phi_R)*np.sin(theta), np.exp(Im*phi_T)*np.cos(theta)],
    #     [np.exp(Im*phi_T)*np.cos(theta), -np.exp(Im*phi_R)*np.sin(theta)]]
    # )
    return np.array(
        [[np.sin(theta), np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]]
    )

def optical_switch(in_a, in_b, transfer_matrix):
    in_ = np.vstack((in_a, in_b))
    out_ = np.einsum('ij,j', transfer_matrix, in_)
    out_c, out_d = out_[:, 0], out_[:, 1]
    return out_c, out_d

def optical_switch1(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(step, 1, np.pi/4)
    return optical_switch(in_a, in_b, transfer_matrix)

def optical_switch2(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(step, 2, np.pi/4)
    return optical_switch(in_a, in_b, transfer_matrix)

def collapse_state(out_c, out_d):
    pass

def main(n_steps, output_dirpath):
    # Relative phase shift between early and late photons #
    delta_phi = np.pi / 4

    in_a, in_b = generate_initial_pulse()
    for step in range(n_steps):
        out_c, out_d = optical_switch1(in_a, in_b, step)
        in_a, in_b = optical_switch2(out_c, out_d, step)
    
    collapse_state(out_c, out_d)

def store(output_data, output_dirpath):
    pass

























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







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate the quantum random walk.'
    )
    parser.add_argument('n-steps', dest='n_steps', action='store', default=50,
        help='The number of steps to iterate the simulation.'
    )
    parser.add_argument('--dump', dest='output_dir_path', action='store', default=f'{str(Path().absolute())}/../output_sim',
        help='Name of the output path in which the processed parquets will be stored.'
    )
    args = parser.parse_args()

    main(args.n_steps, args.output_dir_path)