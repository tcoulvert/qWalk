import numpy as np

N_STEPS = 50

def generate_initial_pulse():
    return (1, 0)

def compute_transfer_matrix(step, switch_num):
    return [
        [0.5, 0.5],
        [0.5, 0.5]
    ]

def optical_switch(in_a, in_b, transfer_matrix):
    in_ = np.vstack((in_a, in_b))
    out_ = np.einsum('ij,j', transfer_matrix, in_)
    out_c, out_d = out_[:, 0], out_[:, 1]
    return out_c, out_d

def optical_switch1(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(step, 1)
    return optical_switch(in_a, in_b, transfer_matrix)

def optical_switch2(in_a, in_b, step):
    transfer_matrix = compute_transfer_matrix(step, 2)
    return optical_switch(in_a, in_b, transfer_matrix)

def collapse_state(out_c, out_d):
    pass

def main():
    # Relative phase shift between early and late photons #
    delta_phi = np.pi / 2

    in_a, in_b = generate_initial_pulse()
    for n in range(N_STEPS):
        out_c, out_d = optical_switch1(in_a, in_b, n)
        in_a, in_b = optical_switch2(out_c, out_d, n)
    
    collapse_state(out_c, out_d)




if __name__ == '__main__':
    main()