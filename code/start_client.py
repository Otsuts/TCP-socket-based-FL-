import subprocess
import time

PREFIX = ['python', 'client.py']
for _ in range(20):
    for number in range(20):
        subprocess.run(PREFIX + [
            f'--client_idx={number + 1}',
        ])
    time.sleep(15)
