import numpy as np

def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    # Write code here
    if current_step < warmup_steps:
        lr = base_lr * (current_step / warmup_steps)
    else:
        decay_steps = total_steps - warmup_steps
        decay_factor = max(0, (decay_steps - (current_step - warmup_steps)) / decay_steps)
        lr = base_lr * decay_factor
    return lr