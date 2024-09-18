import psutil
import time

# Parameters
CHECK_INTERVAL = 1  # in seconds
SPIKE_THRESHOLD = 5  # percent increase that constitutes a spike

def get_memory_usage():
    """Return memory usage in percentage."""
    return psutil.virtual_memory().percent

def detect_memory_spike(prev_usage, current_usage, threshold):
    """Detect if there is a memory usage spike."""
    if prev_usage == 0:
        return False
    increase = ((current_usage - prev_usage) / prev_usage) * 100
    return increase >= threshold

def main():
    previous_usage = get_memory_usage()
    print(f"Initial memory usage: {previous_usage:.2f}%")
    
    while True:
        time.sleep(CHECK_INTERVAL)
        current_usage = get_memory_usage()
        
        if detect_memory_spike(previous_usage, current_usage, SPIKE_THRESHOLD):
            print(f"Memory spike detected! Current usage: {current_usage:.2f}% (Prev: {previous_usage:.2f}%)")
        
        # Update the previous usage for the next check
        previous_usage = current_usage

if __name__ == "__main__":
    main()
