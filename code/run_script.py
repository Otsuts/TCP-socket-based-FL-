import subprocess


PREFIX = ['python', 'main.py']

for NUM_PART in ['16', '17', '18', '19', '20']:
    subprocess.run(PREFIX +
                   [
                       f'--num_part={NUM_PART}',
                       '--method=filewriting'
                   ]
                   )
