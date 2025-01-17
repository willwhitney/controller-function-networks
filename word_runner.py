import os
import sys

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


# networks_dir = '/om/user/wwhitney/facegen_networks/'
base_networks = {
    }


# Don't give it a save name - that gets generated for you
jobs = [
        {
            'model': 'lstm',
            'rnn_size': 600,
            'num_layers': 3,
        },
        {
            'steps_per_output': 1,
            'rnn_size': 256,
            'steps_per_output': 3,
            'num_functions': 400,
        },
        {
            'rnn_size': 400,
            'num_layers': 3,
            'steps_per_output': 3,
            'num_functions': 200,
        },
        {
            'rnn_size': 500,
            'num_layers': 3,
            'steps_per_output': 3,
            'num_functions': 200,
        },
        {
            'rnn_size': 256,
            'num_layers': 3,
            'steps_per_output': 5,
            'num_functions': 300,
        },
        {
            'rnn_size': 256,
            'num_layers': 3,
            'steps_per_output': 5,
            'num_functions': 400,
            'seed': 1,
        },
        {
            'rnn_size': 256,
            'num_layers': 3,
            'steps_per_output': 5,
            'num_functions': 500,
        },
    ]

if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "wiki"
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if job[flag]:
                jobname = jobname + "_" + flag
                flagstring = flagstring + " -" + flag
            else:
                print "WARNING: Excluding 'False' flag " + flag
        elif flag == 'import':
            imported_network_name = job[flag]
            if imported_network_name in base_networks.keys():
                network_location = base_networks[imported_network_name]
                jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                flagstring = flagstring + " -" + flag + " " + str(network_location)
            else:
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " -" + flag + " " + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " -" + flag + " " + str(job[flag])
    flagstring = flagstring + " -name " + jobname

    jobcommand = "th word_main.lua" + flagstring

    print(jobcommand)
    if local and not dry_run:
        if detach:
            os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
        else:
            os.system(jobcommand)

    else:
        with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            slurmfile.write("#!/bin/bash\n")
            slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
            slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
            slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            slurmfile.write(jobcommand)

        if not dry_run:
            if 'gpuid' in job and job['gpuid'] >= 0:
                os.system("sbatch -N 1 -c 2 --gres=gpu:1 -p gpu --mem=8000  --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
            else:
                os.system("sbatch -N 1 -c 6 --mem=10000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
