import subprocess

def get_best_device():
    """
    Determine the GPU device with the largest free memory or fallback to CPU if no GPUs are available.

    Returns:
        best_device (str): The device with the largest free memory (e.g., 'cuda:0') or 'cpu'.
    """
    try:
        # Run `nvidia-smi` and query GPU memory stats
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse the `nvidia-smi` output
        gpu_stats = [
            (f"cuda:{i}", int(free), int(total))
            for i, (free, total) in enumerate(line.split(',') for line in result.stdout.strip().split('\n'))
        ]

        # Display memory stats for all GPUs
        for device, free, total in gpu_stats:
            print(f"Device: {device}, Free Memory: {free} MB, Total Memory: {total} MB")

        # Find GPU with the largest free memory
        if gpu_stats:
            best_device = max(gpu_stats, key=lambda x: x[1])[0]
            print(f"Device with largest free memory: {best_device}")
            return best_device

    except subprocess.CalledProcessError:
        print("Error: Unable to query GPU memory with `nvidia-smi`.")
    except FileNotFoundError:
        print("Error: `nvidia-smi` not found. Make sure NVIDIA drivers are installed.")

    # Fallback to CPU if no GPUs are available or an error occurs
    print("Defaulting to CPU.")
    return "cpu"




def get_top_two_devices():
    """
    Determine the two GPU devices with the largest free memory or fallback to CPU if no GPUs are available.

    Returns:
        top_devices (list): A list of up to two devices with the largest free memory (e.g., ['cuda:0', 'cuda:1']) or ['cpu'].
    """
    try:
        # Run `nvidia-smi` and query GPU memory stats
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE,
            text=True,
            check=True
        )

        # Parse the `nvidia-smi` output
        gpu_stats = [
            (f"cuda:{i}", int(free), int(total))
            for i, (free, total) in enumerate(line.split(',') for line in result.stdout.strip().split('\n'))
        ]

        # Display memory stats for all GPUs
        for device, free, total in gpu_stats:
            print(f"Device: {device}, Free Memory: {free} MB, Total Memory: {total} MB")

        # Sort GPUs by free memory in descending order
        sorted_gpus = sorted(gpu_stats, key=lambda x: x[1], reverse=True)

        # Select up to two GPUs with the largest free memory
        top_devices = [device for device, _, _ in sorted_gpus[:2]]
        if top_devices:
            print(f"Devices with the largest free memory: {top_devices}")
            return top_devices

    except subprocess.CalledProcessError:
        print("Error: Unable to query GPU memory with `nvidia-smi`.")
    except FileNotFoundError:
        print("Error: `nvidia-smi` not found. Make sure NVIDIA drivers are installed.")

    # Fallback to CPU if no GPUs are available or an error occurs
    print("Defaulting to CPU.")
    return ["cpu"]

# Example usage
#top_devices = get_top_two_devices()
#print(f"Selected devices: {top_devices}")


# Example usage
#best_device = get_best_device()
#print(f"Selected device: {best_device}")
