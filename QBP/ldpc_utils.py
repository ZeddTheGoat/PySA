import numpy as np

def ldpc_encode(message, G):
    """
    Encodes a binary message using the generator matrix G.
    """
    if message.ndim != 1:
        raise ValueError("Input message must be a 1D array.")

    codeword = (G @ message) % 2
    return codeword.astype(int)

def add_awgn(symbols, snr_db):
    """
    Adds Additive White Gaussian Noise to modulated symbols.
    """
    snr_linear = 10**(snr_db / 10.0)
    signal_power = np.mean(np.abs(symbols)**2)
    noise_power = signal_power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    
    noise = sigma * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise, sigma
