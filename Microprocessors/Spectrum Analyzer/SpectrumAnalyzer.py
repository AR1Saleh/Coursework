import numpy as np
import matplotlib.pyplot as plt
import serial
from scipy import signal
import time
import sys
import warnings

# Ignore numpy FFT overflow warnings
warnings.filterwarnings('ignore', category=np.ComplexWarning)

# Configuration
SERIAL_PORT = 'COM4'       # Change to your COM port
BAUD_RATE = 1000000        # Must match microcontroller
SAMPLE_RATE = 1000         # 1000 Hz
DURATION = 60              # 60 seconds
TOTAL_SAMPLES = SAMPLE_RATE * DURATION
DATA_SIZE = TOTAL_SAMPLES * 8  # 8 bytes per sample (2 floats)

def receive_data():
    """Capture data from serial port"""
    print(f"Waiting for data on {SERIAL_PORT}...")
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=DURATION+5) as ser:
        # Wait for start trigger
        while True:
            try:
                line = ser.readline().decode().strip()
            except UnicodeDecodeError:
                continue
                
            if "ACQUIRING" in line:
                print("Acquisition started")
                break
        
        # Receive binary data
        data = bytearray()
        start_time = time.time()
        while len(data) < DATA_SIZE:
            bytes_needed = DATA_SIZE - len(data)
            chunk = ser.read(bytes_needed)
            if not chunk:
                # Timeout protection
                if time.time() - start_time > DURATION + 10:
                    print("Timeout waiting for data!")
                    break
                continue
            data.extend(chunk)
            print(f"Received {len(data)/8}/{TOTAL_SAMPLES} samples", end='\r')
        
        print(f"\nReceived {len(data)/8} samples")
    return data

def process_data(raw_data):
    """Convert raw bytes to signals and compute frequency response"""
    # Convert to float32 array
    samples = np.frombuffer(raw_data, dtype=np.float32)
    
    # Verify length
    if len(samples) < TOTAL_SAMPLES * 2:
        print(f"Warning: Expected {TOTAL_SAMPLES*2} points, got {len(samples)}")
        samples = np.pad(samples, (0, TOTAL_SAMPLES*2 - len(samples)), 'constant')
    
    # Separate u and y
    u = samples[::2]      # Chirp signal
    y = samples[1::2]     # System response
    
    # Apply windowing to reduce spectral leakage
    window = np.hamming(len(u))
    u_windowed = u * window
    y_windowed = y * window
    
    # Compute frequency response
    U = np.fft.rfft(u_windowed)
    Y = np.fft.rfft(y_windowed)
    H = Y / U
    
    # Replace infinite/NaN values
    H[np.isinf(H) | np.isnan(H)] = 0
    
    # Frequency axis
    freqs = np.fft.rfftfreq(len(u), 1/SAMPLE_RATE)
    
    # Calculate magnitude and phase
    magnitude = np.abs(H)
    phase = np.angle(H, deg=True)  # Phase in degrees
    phase_unwrapped = np.unwrap(phase)  # Unwrapped phase
    
    return u, y, freqs, magnitude, phase_unwrapped

def plot_results(u, y, freqs, magnitude, phase):
    """Generate comprehensive Bode plots"""
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Time Domain Plot
    time_axis = np.arange(len(u)) / SAMPLE_RATE
    ax1.plot(time_axis, u, 'b-', label='Chirp Signal (u)')
    ax1.plot(time_axis, y, 'r-', label='System Response (y)')
    ax1.set_title('Time Domain Signals')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend()
    ax1.grid(True)
    
    # Magnitude Plot (Bode)
    ax2.semilogx(freqs[1:], 20*np.log10(magnitude[1:]), 'b-')
    ax2.set_title('Frequency Response (Magnitude)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.grid(True, which='both', axis='both')
    ax2.set_xlim(0.1, 100)  # Focus on 0.1-100Hz range
    
    # Phase Plot (Bode)
    ax3.semilogx(freqs[1:], phase[1:], 'r-')
    ax3.set_title('Frequency Response (Phase)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Phase (degrees)')
    ax3.grid(True, which='both', axis='both')
    ax3.set_xlim(0.1, 100)  # Match magnitude plot range
    ax3.set_yticks(np.arange(-360, 361, 45))  # 45° increments
    
    plt.tight_layout()
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'full_bode_plot_{timestamp}.png', dpi=300)
    
    # Save data for future analysis
    np.savez(f'spectrum_data_{timestamp}.npz', 
             u=u, y=y, freqs=freqs, 
             magnitude=magnitude, phase=phase)
    
    print(f"Results saved as full_bode_plot_{timestamp}.png and spectrum_data_{timestamp}.npz")
    plt.show()

def main():
    print("Spectrum Analyzer - Frequency Response Measurement")
    print("==================================================")
    
    # Step 1: Receive data from microcontroller
    raw_data = receive_data()
    
    if len(raw_data) < DATA_SIZE:
        print(f"Error: Only received {len(raw_data)} bytes, expected {DATA_SIZE}")
        return
    
    # Step 2: Process data
    u, y, freqs, magnitude, phase = process_data(raw_data)
    
    print("Data processing complete")
    print(f"Frequency range: {freqs[1]:.2f}Hz to {freqs[-1]:.2f}Hz")
    
    # Step 3: Plot and save results
    plot_results(u, y, freqs, magnitude, phase)

if __name__ == "__main__":
    main()