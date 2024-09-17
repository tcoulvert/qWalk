import argparse
from pathlib import Path

import awkward as ak
import numpy as np

def compute_transfer_matrix(theta, phi_R=0, phi_T=0):
    ## This method will become important once we start doing the actual parton shower simulation
    ##  and need to change the reflectivity of the beam-splitters every cycle.

    # maybe need to multiply front by factor like 'np.exp(Im*phi_R(T))'??
    return np.array(
        [[np.sin(theta), np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]]
    )

def optical_switch(in_a, in_b, transfer_matrix):
    in_ = np.vstack((in_a, in_b))
    out_ = np.einsum('ij,j', transfer_matrix, in_)
    out_c, out_d = out_[:, 0], out_[:, 1]
    return out_c, out_d

def vacuum_noise():
    ## TO BE USED LATER FOR MODELING NOISE ##
    return 0

def input_pump(step):
    ## TO BE USED LATER FOR PUMPING EXPERIMENT ##
    if step == 1:  # index step from 1 so it matches with 'number of input pulses'
        return np.array([1])
    else:
        return vacuum_noise() * np.ones(step)
    
def switch1_scheduler(step):
    ## TO BE USED LATER WHEN ACTUALLY MODELLING PARTON SHOWER ##
    return np.pi/4
    
def optical_switch1(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(1, switch1_scheduler(step))
    return optical_switch(in_a, in_b, transfer_matrix)

def optical_switch2(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(1, np.pi/4)
    return optical_switch(in_a, in_b, transfer_matrix)

def main(n_steps, output_dirpath):
    # Relative phase shift between early and late photons #
    delta_phi = np.pi / 2

    pump_array = ak.Array()
    detector_array = ak.Array()
    feedback_array = ak.Array()
    feedback_array['step_1'] = np.array([vacuum_noise()])
    for step in range(1, n_steps+1):
        pump_array[f'step_{step}'] = input_pump(step)

        early_c, late_d = np.zeros(step), np.zeros(step)
        for pulse in range(len(pump_array[f'step_{step}'])):
            early_c[pulse], late_d[pulse] = optical_switch1(
                pump_array[f'step_{step}'][pulse],
                feedback_array[f'step_{step}'][pulse],
                step
            )
        rectified_c = np.concatenate(early_c, np.array([vacuum_noise()]))
        rectified_d = np.concatenate(np.array([vacuum_noise()]), late_d)

        feedback_array[f'step_{step+1}'] = np.zeros(step+1)
        detector_array[f'step_{step+1}'] = np.zeros(step+1)
        for pulse, (pulse_a, pulse_b) in enumerate(zip(rectified_c, rectified_d)):
            (
                feedback_array[f'step_{step+1}'][pulse], 
                detector_array[f'step_{step+1}'][pulse]
            ) = optical_switch2(pulse_a, pulse_b, step)











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