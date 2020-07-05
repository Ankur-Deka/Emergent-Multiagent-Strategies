import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# numAttackers = 5
# numGuards = 5
# csvPath = './marlsave/save_new/tmp_11-(copy)/logs/csv/data.csv'

# data = pd.read_csv(csvPath, sep=',').values

# time = data[:,0]
# guardRewards = data[:,1:1+numGuards].T
# attackerRewards = data[:,1+numGuards:1+numGuards+numAttackers].T


# print(guardRewards.shape, time.shape)

# # plt.subplot(2,1,1)
# sns.set(style="darkgrid", font_scale=1.5)
# sns.tsplot(data=guardRewards, time=time, color='green', ci='sd')
# # plt.subplot(2,1,2)
# # sns.set(style="darkgrid", font_scale=1.5)
# sns.tsplot(data=attackerRewards, time=time, color='red', ci='sd')
# plt.show()

# # arr = np.load('data.npy')
# # print(arr.shape)

load_dir = './marlsave/save_new/tmp_16'
arr = np.load(os.path.join(load_dir,'reward_data_ensemble.npy'))
print(arr.shape)

ckpts = arr[:,0,0]*1000
sigma = 2
guardRewards = gaussian_filter1d(arr[:,:,1:6].reshape(arr.shape[0],-1).T, sigma)
attackerRewards = gaussian_filter1d(arr[:,:,6:11].reshape(arr.shape[0],-1).T, sigma)


w = 15
h = 10
d = 100
plt.figure(figsize=(w, h), dpi=d)


# plt.ticklabel_format(style='plain', axis='x',useOffset=False)
sns.set(style="darkgrid", font_scale=1.5)
ax = sns.tsplot(data=guardRewards, time=ckpts, color='green', ci='sd', linewidth=4)
# ax = sns.tsplot(data=attackerRewards, time=ckpts, color='red', ci='sd', linewidth=4)
ax.set(xlabel='Environment steps', ylabel='Reward')
plt.xlim(-1e5, None)
plt.tight_layout()
# sns.set(style="darkgrid", font_scale=1.5)

plt.savefig('ensemble reward guard')