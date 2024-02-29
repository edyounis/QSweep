import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('experiment_data.pkl', 'rb') as f:
    data = pickle.load(f)

ds = [2, 3, 4, 5]

cbc_times = [np.mean(data['cbc'][d]['times']) for d in ds]
cbc_pulses = [np.mean(data['cbc'][d]['num_pulses']) for d in ds]
cbc_pulse_std = [np.std(data['cbc'][d]['num_pulses']) for d in ds]
cbc_time_stds = [np.std(data['cbc'][d]['times']) for d in ds]

rbr_times = [np.mean(data['rbr'][d]['times']) for d in ds]
rbr_pulses = [np.mean(data['rbr'][d]['num_pulses']) for d in ds]
rbr_pulse_std = [np.std(data['rbr'][d]['num_pulses']) for d in ds]
rbr_time_stds = [np.std(data['rbr'][d]['times']) for d in ds]

qsweep_times = [np.mean(data['qsweep'][d]['times']) for d in ds]
qsweep_pulses = [np.mean(data['qsweep'][d]['num_pulses']) for d in ds]
qsweep_pulse_std = [np.std(data['qsweep'][d]['num_pulses']) for d in ds]
qsweep_time_stds = [np.std(data['qsweep'][d]['times']) for d in ds]

qsearch_times = [np.mean(data['qsearch'][d]['times']) for d in ds]
qsearch_pulses = [np.mean(data['qsearch'][d]['num_pulses']) for d in ds]
qsearch_pulse_std = [np.std(data['qsearch'][d]['num_pulses']) for d in ds]
qsearch_time_stds = [np.std(data['qsearch'][d]['times']) for d in ds]

print("!"*40)

print(np.mean([np.mean(data['cbc'][d]['num_pulses']) - np.mean(data['qsweep'][d]['num_pulses']) for d in ds]))
print(np.mean([np.mean(data['rbr'][d]['num_pulses']) - np.mean(data['qsweep'][d]['num_pulses']) for d in ds]))
print(np.mean([np.mean(data['qsearch'][d]['num_pulses']) - np.mean(data['qsweep'][d]['num_pulses']) for d in ds[:2]]))

print(np.mean(data['qsweep'][5]['times']))

print("!"*40)

# print(cbc_times)
# print(cbc_pulses)
# print(cbc_time_stds)

# print(rbr_times)
# print(rbr_pulses)
# print(rbr_time_stds)

# print(qsweep_times)
# print(qsweep_pulses)
# print(qsweep_time_stds)

# print(qsearch_times)
# print(qsearch_pulses)
# print(qsearch_time_stds)

plt.style.use('ggplot')
fig, axs = plt.subplots(nrows=1, ncols=2, dpi=600, figsize=(8, 4), tight_layout=True)
ax, ax3 = axs

ax.plot(ds, rbr_times, label='RBR', marker='d')
ax.plot(ds, cbc_times, label='CBC', marker='d')
ax.plot(ds, qsweep_times, label='QSweep', marker='d')
ax.plot(ds[:2], qsearch_times[:2], label='QSearch', marker='d')
# ax.fill_between(ds, np.array(cbc_times) - np.array(cbc_time_stds), np.array(cbc_times) + np.array(cbc_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(rbr_times) - np.array(rbr_time_stds), np.array(rbr_times) + np.array(rbr_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(qsweep_times) - np.array(qsweep_time_stds), np.array(qsweep_times) + np.array(qsweep_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(qsearch_times) - np.array(qsearch_time_stds), np.array(qsearch_times) + np.array(qsearch_time_stds), alpha=0.2)
ax.set_yscale('log')
ax.set_title('Clifford Synthesis')
ax.set_yscale('log')
ax.set_ylabel('Time (s)')
ax.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax.set_ylim(0.0001, 2000)
ax.set_xlabel('Qudit Dimension (d)')
ax.set_xticks(ds)
# ax.legend()



ax2 = ax.twinx()
ax2.grid(False)
ax2.plot(ds, rbr_pulses, label='RBR', linestyle='--', marker='o', markersize=6, linewidth=1.8)
ax2.plot(ds, cbc_pulses, label='CBC', linestyle='--', marker='o', markersize=6, linewidth=2)
ax2.plot(ds, qsweep_pulses, label='QSweep', linestyle='--', marker='o', markersize=6, linewidth=1.6)
ax2.plot(ds[:2], qsearch_pulses[:2], label='QSearch', linestyle='--', marker='o', markersize=6, linewidth=2)
# ax2.fill_between(ds, np.array(cbc_pulses) - np.array(cbc_pulse_std), np.array(cbc_pulses) + np.array(cbc_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(rbr_pulses) - np.array(rbr_pulse_std), np.array(rbr_pulses) + np.array(rbr_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(qsweep_pulses) - np.array(qsweep_pulse_std), np.array(qsweep_pulses) + np.array(qsweep_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(qsearch_pulses) - np.array(qsearch_pulse_std), np.array(qsearch_pulses) + np.array(qsearch_pulse_std), alpha=0.2)
# ax2.set_xlabel('Qudit Dimension (d)')
ax2.set_xticks(ds)
# ax2.set_ylabel('Number of Pulses')
ax2.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ax2.set_yticklabels([''] * 11)


with open('../2_random_haar/experiment_data.pkl', 'rb') as f:
    haar_data = pickle.load(f)

cbc_times = [np.mean(haar_data['cbc'][d]['times']) for d in ds]
cbc_pulses = [np.mean(haar_data['cbc'][d]['num_pulses']) for d in ds]
cbc_pulse_std = [np.std(haar_data['cbc'][d]['num_pulses']) for d in ds]
cbc_time_stds = [np.std(haar_data['cbc'][d]['times']) for d in ds]

rbr_times = [np.mean(haar_data['rbr'][d]['times']) for d in ds]
rbr_pulses = [np.mean(haar_data['rbr'][d]['num_pulses']) for d in ds]
rbr_pulse_std = [np.std(haar_data['rbr'][d]['num_pulses']) for d in ds]
rbr_time_stds = [np.std(haar_data['rbr'][d]['times']) for d in ds]

qsweep_times = [np.mean(haar_data['qsweep'][d]['times']) for d in ds]
qsweep_pulses = [np.mean(haar_data['qsweep'][d]['num_pulses']) for d in ds]
qsweep_pulse_std = [np.std(haar_data['qsweep'][d]['num_pulses']) for d in ds]
qsweep_time_stds = [np.std(haar_data['qsweep'][d]['times']) for d in ds]

qsearch_times = [np.mean(haar_data['qsearch'][d]['times']) for d in ds]
qsearch_pulses = [np.mean(haar_data['qsearch'][d]['num_pulses']) for d in ds]
qsearch_pulse_std = [np.std(haar_data['qsearch'][d]['num_pulses']) for d in ds]
qsearch_time_stds = [np.std(haar_data['qsearch'][d]['times']) for d in ds]

ax3.plot(ds, rbr_times, label='RBR', marker='d')
ax3.plot(ds, cbc_times, label='CBC', marker='d')
ax3.plot(ds, qsweep_times, label='QSweep', marker='d')
ax3.plot(ds[:2], qsearch_times[:2], label='QSearch', marker='d')
# ax.fill_between(ds, np.array(cbc_times) - np.array(cbc_time_stds), np.array(cbc_times) + np.array(cbc_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(rbr_times) - np.array(rbr_time_stds), np.array(rbr_times) + np.array(rbr_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(qsweep_times) - np.array(qsweep_time_stds), np.array(qsweep_times) + np.array(qsweep_time_stds), alpha=0.2)
# ax.fill_between(ds, np.array(qsearch_times) - np.array(qsearch_time_stds), np.array(qsearch_times) + np.array(qsearch_time_stds), alpha=0.2)
ax3.set_yscale('log')
ax3.set_title('Haar Synthesis')
ax3.set_yscale('log')
# ax3.set_ylabel('Time (s)')

ax3.set_yticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax3.set_yticklabels([''] * 7)
ax3.set_ylim(0.0001, 2000)
ax3.set_xlabel('Qudit Dimension (d)')
ax3.set_xticks(ds)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Patch(facecolor='#E24A33', edgecolor='#E24A33', label='RBR'),
                     Patch(facecolor='#348ABD', edgecolor='#348ABD', label='CBC'),
                     Patch(facecolor='#988ED5', edgecolor='#988ED5', label='QSweep'),
                     Patch(facecolor='#777777', edgecolor='#777777', label='QSearch'),
                     Line2D([0], [0], color='black', linestyle='-', label='Time'),
                     Line2D([0], [0], color='black', linestyle='--', label='Pulses'),]
ax3.legend(handles=legend_elements, loc='upper left', prop={'size': 11}, ncol=2, bbox_to_anchor=(-0.01, 0.91, 0.5, 0.10))



ax4 = ax3.twinx()
ax4.grid(False)
ax4.plot(ds, rbr_pulses, label='RBR', linestyle='--', marker='o', markersize=6, linewidth=1.8)
ax4.plot(ds, cbc_pulses, label='CBC', linestyle='--', marker='o', markersize=6, linewidth=2)
ax4.plot(ds, qsweep_pulses, label='QSweep', linestyle='--', marker='o', markersize=6, linewidth=1.6)
ax4.plot(ds[:1], qsearch_pulses[:1], label='QSearch', linestyle='--', marker='o', markersize=6, linewidth=2)
# ax2.fill_between(ds, np.array(cbc_pulses) - np.array(cbc_pulse_std), np.array(cbc_pulses) + np.array(cbc_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(rbr_pulses) - np.array(rbr_pulse_std), np.array(rbr_pulses) + np.array(rbr_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(qsweep_pulses) - np.array(qsweep_pulse_std), np.array(qsweep_pulses) + np.array(qsweep_pulse_std), alpha=0.2)
# ax2.fill_between(ds, np.array(qsearch_pulses) - np.array(qsearch_pulse_std), np.array(qsearch_pulses) + np.array(qsearch_pulse_std), alpha=0.2)
# ax4.set_xlabel('Qudit Dimension (d)')
ax4.set_xticks(ds)
ax4.set_ylabel('Number of Pulses')
ax4.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# fig.suptitle('Qudit Dimension (d)')
fig.savefig('graph.pdf')

# def set_box_color(bp, color):
#     plt.setp(bp['boxes'], color=color)
#     plt.setp(bp['whiskers'], color=color)
#     plt.setp(bp['caps'], color=color)
#     plt.setp(bp['medians'], color=color)
#     plt.setp(bp['fliers'], color=color)

# color1 = "#D7191C"
# color2 = "#2C7BB6"
# color3 = "#F0E442"
# color4 = "#4DAF4A"
# plt.style.use('ggplot')
# fig, ax = plt.subplots(dpi=600)
# cbc_box = ax.boxplot([data['cbc'][d]['times'] for d in ds], vert=True, patch_artist=True, positions=np.array(np.arange(4))*2.0-0.8, labels=ds, flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markerfacecolor': color1})
# rbr_box = ax.boxplot([data['rbr'][d]['times'] for d in ds], vert=True, patch_artist=True, positions=np.array(np.arange(4))*2.0-0.3, labels=ds, flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markerfacecolor': color2})
# qsweep_box = ax.boxplot([data['qsweep'][d]['times'] for d in ds], vert=True, patch_artist=True, positions=np.array(np.arange(4))*2.0+0.3, labels=ds, flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markerfacecolor': color3})
# qsearch_box = ax.boxplot([data['qsearch'][d]['times'] for d in ds], vert=True, patch_artist=True, positions=np.array(np.arange(4))*2.0+0.8, labels=ds, flierprops={'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markerfacecolor': color4})
# set_box_color(cbc_box, color1)
# set_box_color(rbr_box, color2)
# set_box_color(qsweep_box, color3)
# set_box_color(qsearch_box, color4)

# ax.set_yscale('log')
# ax.set_title('Random Clifford Synthesis Time')
# ax.set_yscale('log')
# ax.set_ylabel('Time (s)')
# ax.set_xlabel('Qudit Dimension (d)')
# ax.set_xticks(ds)
# ax.legend()
# fig.savefig('graph.png')
