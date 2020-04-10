import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from arguments import get_args


if __name__=='__main__':

    # Set the folder name depending on experiment
    # folder_name = '/0405_200ep_fs'
    folder_name = '/0505_200ep_fs_8ncpu'
    
    # Load arguments and rewards
    args = get_args()
    rewards_path = args.save_dir + args.env_name + folder_name + '/rewards.npz'
    rewards = np.load(rewards_path)

    # Extract success per epoch
    epochs = rewards['ep']
    succ_rates = rewards['suc']
    assert(epochs.shape==succ_rates.shape)
    
    # Plot
    plt.plot(epochs, succ_rates)
    plt.show()
    print("Executed")


