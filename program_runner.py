import os
import sys

dry_run = '--dry-run' in sys.argv
local   = '--local' in sys.argv
detach  = '--detach' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")


networks_prefix = "networks"

# networks_dir = '/om/user/wwhitney/facegen_networks/'
base_networks = {
        "onestep": "networks/trained_onestep/trained.t7"
    }


# Don't give it a save name - that gets generated for you
# jobs = [
#         {
#             "import": "onestep",
#         },
#
#
#     ]

jobs = []

noise_options = [0, 0.01, 0.1]
sharpening_rate_options = [0, 1, 10]
lr_decay_options = [1, 0.999, 0.97]
function_lr_options = [1e-3, 1e-2, 2e-2]

# jobs with feedforward controller, one metadata step at a time
for noise in noise_options:
    for sharpening_rate in sharpening_rate_options:
        for lr_decay in lr_decay_options:
            for function_lr in function_lr_options:
                job = {
                        "model": "ff-controller",
                        "noise": noise,
                        "sharpening_rate": sharpening_rate,
                        "learning_rate_decay": lr_decay,
                        "function_learning_rate": function_lr,
                    }
                jobs.append(job)

# jobs = []
#
# noise_options = [0, 0.01, 0.03, 0.1]
# sharpening_rate_options = [0, 1, 5, 10]
# lr_decay_options = [1, 0.995, 0.99, 0.97]
# function_lr_options = [0, 1e-4, 1e-3, 1e-2]
#
# # make jobs without pretrained functions
# for noise in noise_options:
#     for sharpening_rate in sharpening_rate_options:
#         for lr_decay in lr_decay_options:
#             job = {
#                     "noise": noise,
#                     "sharpening_rate": sharpening_rate,
#                     "learning_rate_decay": lr_decay,
#                 }
#             jobs.append(job)
#
# # make jobs with pretrained functions
# for noise in noise_options:
#     for sharpening_rate in sharpening_rate_options:
#         for lr_decay in lr_decay_options:
#             for function_lr in function_lr_options:
#                 job = {
#                         "import": "onestep",
#                         "noise": noise,
#                         "sharpening_rate": sharpening_rate,
#                         "learning_rate_decay": lr_decay,
#                         "function_learning_rate": function_lr,
#                     }
#                 jobs.append(job)



if dry_run:
    print "NOT starting jobs:"
else:
    print "Starting jobs:"

for job in jobs:
    jobname = "onestep"
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
                flagstring = flagstring + " -" + flag + " " + networks_prefix + "/" + str(job[flag])
        else:
            jobname = jobname + "_" + flag + "_" + str(job[flag])
            flagstring = flagstring + " -" + flag + " " + str(job[flag])
    flagstring = flagstring + " -name " + jobname

    jobcommand = "th program_main.lua" + flagstring

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
                os.system("sbatch -N 1 -c 2 --gres=gpu:1 -p gpu --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
            else:
                os.system("sbatch -N 1 -c 2 --mem=8000 --time=6-23:00:00 slurm_scripts/" + jobname + ".slurm &")
