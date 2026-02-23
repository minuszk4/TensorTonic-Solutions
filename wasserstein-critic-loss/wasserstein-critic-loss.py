import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    # Write code here
    E_real = np.mean(real_scores)
    E_fake = np.mean(fake_scores)
    loss = E_fake - E_real
    return loss