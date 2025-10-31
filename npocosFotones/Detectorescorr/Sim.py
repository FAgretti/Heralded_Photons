import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_windows = 10000  # Number of integration windows
window_duration = 1e-8  # Integration window duration (seconds)
rate = 1e12  # Average photon rate (Hz)
antibunching = True  # If True, simulate antibunched source

# Function to generate photon arrival times
def generate_photon_arrivals(rate, total_time, antibunching=False):
    if antibunching:
        # For antibunched source, use exponential waiting time with minimum gap
        min_gap = 1e-10  # Minimum gap between photons (seconds)
        times = []
        t = 0
        while t < total_time:
            dt = np.random.exponential(1/rate)
            if dt < min_gap:
                dt = min_gap
            t += dt
            if t < total_time:
                times.append(t)
        return np.array(times)
    else:
        # Poissonian source
        n_photons = np.random.poisson(rate * total_time)
        return np.sort(np.random.uniform(0, total_time, n_photons))

# Simulate experiment
total_time = n_windows * window_duration
photon_times = generate_photon_arrivals(rate, total_time, antibunching=antibunching)

# Simulate beam splitter (50/50)
split_probs = np.random.rand(len(photon_times))
# Each photon has a 50% chance to go to each detector, but allow for both detectors to register photons in the same window (classical APD)
det1_times = photon_times[split_probs < 0.5]
det2_times = photon_times[split_probs >= 0.5]
# For classical APDs, add a small cross-talk: some photons are detected by both detectors
cross_talk = 0.1  # 5% chance a photon is detected by both
both_detected = photon_times[np.random.rand(len(photon_times)) < cross_talk]
det1_times = np.concatenate([det1_times, both_detected])
det2_times = np.concatenate([det2_times, both_detected])

# Bin photon arrivals into windows
bins = np.arange(0, total_time + window_duration, window_duration)
I1, _ = np.histogram(det1_times, bins)
I2, _ = np.histogram(det2_times, bins)

# Calculate g2(0)
g2_num = np.mean(I1 * I2)
g2_den = np.mean(I1) * np.mean(I2)
g2 = g2_num / g2_den if g2_den > 0 else np.nan

print(f"g2(0) = {g2:.3f}")

# Plot histogram of intensities
plt.figure(figsize=(8,4))
plt.hist(I1, bins=20, alpha=0.5, label='Detector 1')
plt.hist(I2, bins=20, alpha=0.5, label='Detector 2')
plt.xlabel('Photons per window')
plt.ylabel('Count')
plt.legend()
plt.title(f'Intensity histograms, g2(0)={g2:.3f}')
plt.tight_layout()
plt.show()
