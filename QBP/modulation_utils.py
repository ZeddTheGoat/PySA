import numpy as np

def get_modulation_params(modulation_type):
    """Returns the number of bits per symbol."""
    if modulation_type == 'BPSK': return 1
    if modulation_type == 'QPSK': return 2
    if modulation_type == '16QAM': return 4
    raise ValueError(f"Unsupported modulation type: {modulation_type}")

def map_binary_to_symbol(binary_bits, modulation_type):
    """Maps a block of binary bits to a complex constellation symbol."""
    spin_map = (1 - 2 * np.array(binary_bits))
    if modulation_type == 'BPSK': return spin_map[0]
    if modulation_type == 'QPSK': return (spin_map[0] + 1j * spin_map[1]) / np.sqrt(2)
    if modulation_type == '16QAM':
        real = 2 * spin_map[0] + spin_map[1]
        imag = 2 * spin_map[2] + spin_map[3]
        return (real + 1j * imag) / np.sqrt(10)
    raise ValueError(f"Unsupported modulation type: {modulation_type}")


def calculate_llrs(received_symbols, noise_variance, modulation_type, n_codeword):
    """
    Calculates the Log-Likelihood Ratio (LLR) for each bit.

    For higher-order modulations (e.g. 16QAM), when a symbol includes padding bits
    (i.e. some of the symbol's bit positions are beyond the original codeword length),
    those padding bit positions are treated as fixed zeros and candidate constellation
    points are restricted accordingly when computing per-bit LLRs.
    """
    bits_per_symbol = get_modulation_params(modulation_type)
    num_symbols = len(received_symbols)
    total_bits = num_symbols * bits_per_symbol
    llrs = np.zeros(total_bits)

    if modulation_type == 'BPSK':
        for i in range(num_symbols):
            if i < n_codeword:
                llrs[i] = 4 * received_symbols[i].real / noise_variance
        return llrs[:n_codeword]

    if modulation_type == 'QPSK':
        scaling = 4 * (1/np.sqrt(2)) / noise_variance
        for i in range(num_symbols):
            if i*2 < n_codeword: llrs[i*2] = scaling * received_symbols[i].real
            if i*2 + 1 < n_codeword: llrs[i*2 + 1] = scaling * received_symbols[i].imag
        return llrs[:n_codeword]

    # For 16QAM and other higher-order constellations, build constellation list
    constellation = {}
    for i in range(16):
        bits = tuple(int(b) for b in f'{i:04b}'[::-1])
        constellation[bits] = map_binary_to_symbol(bits, modulation_type)

    const_items = list(constellation.items())
    const_keys = [bits for bits, _ in const_items]
    points = np.array([sym for _, sym in const_items])

    for i in range(num_symbols):
        y = received_symbols[i]
        distances_sq = np.abs(y - points)**2

        # determine which bit positions inside this symbol are padding (fixed to 0)
        bit_indices = [i * bits_per_symbol + b for b in range(bits_per_symbol)]
        fixed_positions = [b for b, idx in enumerate(bit_indices) if idx >= n_codeword]

        for k in range(bits_per_symbol):
            bit_index = bit_indices[k]
            if bit_index >= n_codeword:
                # this is a padding bit; skip LLR computation
                continue

            # Filter constellation points to those matching padding bits == 0
            filtered = []
            for j, bits in enumerate(const_keys):
                ok = True
                for p in fixed_positions:
                    if bits[p] != 0:
                        ok = False
                        break
                if ok:
                    filtered.append((j, distances_sq[j], bits))

            # If filtering removed all candidates (shouldn't happen), fall back to unfiltered sets
            if len(filtered) == 0:
                set0_distances = [distances_sq[j] for j, bits in enumerate(const_keys) if bits[k] == 0]
                set1_distances = [distances_sq[j] for j, bits in enumerate(const_keys) if bits[k] == 1]
            else:
                set0_distances = [d for j, d, bits in filtered if bits[k] == 0]
                set1_distances = [d for j, d, bits in filtered if bits[k] == 1]
                # If one side is empty after filtering, fall back to unfiltered for that side
                if len(set0_distances) == 0:
                    set0_distances = [distances_sq[j] for j, bits in enumerate(const_keys) if bits[k] == 0]
                if len(set1_distances) == 0:
                    set1_distances = [distances_sq[j] for j, bits in enumerate(const_keys) if bits[k] == 1]

            llrs[bit_index] = (np.min(set1_distances) - np.min(set0_distances)) / noise_variance

    return llrs[:n_codeword]
