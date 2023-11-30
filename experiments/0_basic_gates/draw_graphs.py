import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('experiment_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Plotting time
x = ['xx', 'yy', 'zz', 'qft', 'cx', 'cy', 'cz', 'ch', 'swap']
x_nums = np.array([i for i in range(len(x))])

rbr = []
cbc = []
qsweep = []
for benchmark in x:
    rbr.append(data['rbr']['times'][benchmark])
    cbc.append(data['cbc']['times'][benchmark])
    qsweep.append(data['qsweep']['times'][benchmark])

qsearch = []
qsearch_data = data['qsearch']['times']
timeouts = []
for benchmark in x:
    if benchmark in qsearch_data:
        qsearch.append(qsearch_data[benchmark])
    else:
        qsearch.append(3600)
        timeouts.append(benchmark)


plt.style.use('ggplot')
fig, ax = plt.subplots(dpi=600)
bar_width = 0.2
ax.bar(x_nums - 1.5*bar_width, rbr, label='RBR', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums - 0.5*bar_width, cbc, label='CBC', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums + 0.5*bar_width, qsweep, label='QSweep', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums + 1.5*bar_width, qsearch, label='QSearch', width=bar_width, linewidth=0.5, edgecolor='black')
ax.set_xticks(x_nums, x)
ax.set_ylabel('Time (s)')
ax.set_yscale('log')
ax.set_title('Single-Ququart Gate Synthesis Time')
for timeout in timeouts:
    ax.text(x_nums[x.index(timeout)] + 1.5*bar_width, 3600, 'TO', ha='center', va='bottom')


timediff1 = '%.0fms' % ((qsweep[x.index('cy')] - cbc[x.index('cy')]) * 1000)
ax.vlines(x_nums[x.index('cy')] - 0.5*bar_width, cbc[x.index('cy')] + 0.0001, qsweep[x.index('cy')] - 0.008, color='black', linewidth=0.5)
ax.hlines(cbc[x.index('cy')] + 0.0001, x_nums[x.index('cy')] - 0.8*bar_width, x_nums[x.index('cy')] - 0.2*bar_width, color='black', linewidth=0.5)
ax.hlines(qsweep[x.index('cy')] - 0.008, x_nums[x.index('cy')] - 0.8*bar_width, x_nums[x.index('cy')] - 0.2*bar_width, color='black', linewidth=0.5)
ax.text(x_nums[x.index('cy')] - 1.1*bar_width, 0.2*(cbc[x.index('cy')] + qsweep[x.index('cy')]), timediff1, ha='center', va='center', rotation=90)


timediff2 = '%.0fs' % ((qsearch[x.index('cy')] - qsweep[x.index('cy')]))
ax.vlines(x_nums[x.index('cy')] + 0.5*bar_width, qsweep[x.index('cy')] + 0.008, qsearch[x.index('cy')] - 200, color='black', linewidth=0.5)
ax.hlines(qsweep[x.index('cy')] + 0.008, x_nums[x.index('cy')] + 0.8*bar_width, x_nums[x.index('cy')] + 0.2*bar_width, color='black', linewidth=0.5)
ax.hlines(qsearch[x.index('cy')] - 200, x_nums[x.index('cy')] + 0.8*bar_width, x_nums[x.index('cy')] + 0.2*bar_width, color='black', linewidth=0.5)
ax.text(x_nums[x.index('cy')] - 0.18*bar_width, 0.005*(qsearch[x.index('cy')] + qsweep[x.index('cy')]), timediff2, ha='center', va='center', rotation=90)

ax.legend()
fig.savefig('time.png')


# Plotting quality
rbr = []
cbc = []
qsweep = []
for benchmark in x:
    rbr.append(data['rbr']['num_pulses'][benchmark])
    cbc.append(data['cbc']['num_pulses'][benchmark])

    if data['qsweep']['num_pulses'][benchmark] == 0:
        qsweep.append(0.05)
    else:
        qsweep.append(data['qsweep']['num_pulses'][benchmark])

qsearch = []
qsearch_data = data['qsearch']['num_pulses']
timeouts = []
for benchmark in x:
    if benchmark in qsearch_data:
        if qsearch_data[benchmark] == 0:
            qsearch.append(0.05)
        else:
            qsearch.append(qsearch_data[benchmark])
    else:
        qsearch.append(0)
        timeouts.append(benchmark)


fig, ax = plt.subplots(dpi=600)
bar_width = 0.2
ax.bar(x_nums - 1.5*bar_width, rbr, label='RBR', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums - 0.5*bar_width, cbc, label='CBC', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums + 0.5*bar_width, qsweep, label='QSweep', width=bar_width, linewidth=0.5, edgecolor='black')
ax.bar(x_nums + 1.5*bar_width, qsearch, label='QSearch', width=bar_width, linewidth=0.5, edgecolor='black')
ax.set_xticks(x_nums, x)
ax.set_ylabel('Pulses in Decomposition')
ax.set_title('Single-Ququart Gate Synthesis Quality')
for timeout in timeouts:
    ax.text(x_nums[x.index(timeout)] + 1.65*bar_width, 0, 'TO', ha='center', va='bottom', fontsize=6)
# ax.legend()
fig.savefig('quality.png')