# import numpy as np
# from matplotlib import pyplot as plt
import seaborn

import sys
import os
import copy
import pprint
from matplotlib import gridspec

import argparse

parser = argparse.ArgumentParser(description='Plot dem results.')
parser.add_argument('--name', default='default')
parser.add_argument('--keep_losers', default=False)
parser.add_argument('--loser_threshold', default=1)
args = parser.parse_args()

output_dir = "reports/" + args.name

pp = pprint.PrettyPrinter(indent=4)

def mean(l):
    return sum(l) / float(len(l))

networks = {}
for name in sys.stdin:
    network_name = name.strip()
    # print(network_name)
    opt_path = "networks/" + network_name + "/opt.txt"
    loss_path = "networks/" + network_name + "/val_loss.txt"
    # print(os.path.isfile(opt_path))
    # print(os.path.isfile(loss_path))
    try:
        if os.path.isfile(opt_path) and os.path.isfile(loss_path):
            network_data = {}
            with open(opt_path) as opt_file:
                options = {}
                for line in opt_file:
                    k, v = line.split(": ")
                    options[k] = v.strip()
                network_data['options'] = options
                network_data['options']['name'] = network_name

            with open(loss_path) as loss_file:
                losses = []
                for line in loss_file:
                    losses.append(float(line))
                network_data['losses'] = losses

            networks[network_name] = network_data

    except IOError as e:
        pass

network_ages = []
for network_name in networks:
    network = networks[network_name]
    network_ages.append(len(network['losses']))

mean_network_age = mean(network_ages)

new_networks = {}
for network_name in networks:
    network = networks[network_name]
    if len(network['losses']) < (3 * mean_network_age / 4.):
        print("Network is too young. Excluding: " + network_name)
    else:
        new_networks[network_name] = network

networks = new_networks

if not args.keep_losers:
    new_networks = {}
    for network_name in networks:
        network = networks[network_name]
        if network['losses'] > args.loser_threshold:
            print("Network's loss is too high. Excluding: " + network_name)
        else:
            new_networks[network_name] = network

    networks = new_networks

same_options = copy.deepcopy(networks[networks.keys()[0]]['options'])
diff_options = []
for network_name in networks:
    network = networks[network_name]
    options = network['options']
    for option in options:
        if option not in diff_options:
            if option not in same_options:
                diff_options.append(option)
            else:
                if options[option] != same_options[option]:
                    diff_options.append(option)
                    same_options.pop(option, None)

print(diff_options)
# print(same_options)

# don't separate them by name
# diff_options.remove("name")

per_option_loss_lists = {}

for option in diff_options:
    option_loss_lists = {}
    for network_name in networks:
        network = networks[network_name]

        option_value = 'none'
        if option in network['options'] and network['options'][option] != '':
            option_value = network['options'][option]

        if option_value not in option_loss_lists:
            option_loss_lists[option_value] = []

        option_loss_lists[option_value].append(network['losses'])

    per_option_loss_lists[option] = option_loss_lists


per_option_mean_losses = {}
for option in per_option_loss_lists:
    per_value_mean_losses = {}
    for option_value in per_option_loss_lists[option]:
        loss_lists = per_option_loss_lists[option][option_value]

        last_losses = [losses[-1] for losses in loss_lists]
        mean_loss = mean(last_losses)
        per_value_mean_losses[option_value] = mean_loss

    per_option_mean_losses[option] = per_value_mean_losses

# pp.pprint(per_option_mean_losses)

lengths = [len(per_option_mean_losses[option]) for option in per_option_mean_losses]
gs = gridspec.GridSpec(1, len(per_option_mean_losses), width_ratios=lengths)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for option in per_option_mean_losses:
    option_values = per_option_mean_losses[option].keys()
    # print([value for value in option_values])
    option_value_losses = [per_option_mean_losses[option][value] for value in option_values]

    # print(option_values)
    # print(option_value_losses)
    fig = seaborn.plt.figure(figsize=(30,15))
    fig.add_subplot()

    # fig.subplots_adjust(right = 1000)
    # fig.subplots_adjust(top = 1000000)
    # fig.subplots_adjust(left = 0)
    # fig.subplots_adjust(bottom = -100000)

    g = seaborn.barplot(x=option_values, y=option_value_losses)
    g.set(title=option)
    g.set_xticklabels(option_values, rotation=25, ha='right')
    g.set_yscale('log')

    # fig.subplots_adjust(right = 1000)
    # fig.subplots_adjust(top = 1000000)
    # fig.subplots_adjust(left = 0)
    # fig.subplots_adjust(bottom = -100000)

    # seaborn.plt.title(option)
    # seaborn.plt
    # fig.subplots_adjust(right = 1)
    # fig.subplots_adjust(left = 0)
    seaborn.plt.tight_layout()

    seaborn.plt.savefig(output_dir + "/" + option + ".png")
    seaborn.plt.close()
    # fig.savefig("report_" + option + ".png")

    # pl.set_xticklabels(rotation=30)
    # axes.bar(range(len(option_values)), option_value_losses)
    # plt.title(option)
    # plt.plot(option_values, option_value_losses)



# fig
# seaborn.plt.show()
