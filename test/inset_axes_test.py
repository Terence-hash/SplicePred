# -*- coding: utf-8 -*-
# @Time    : 2022/7/7 21:31
# @Author  : Terence Tan
# @Email   : 2228254095@qq.com
# @FileName: inset_axes_test.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes, zoomed_inset_axes

x = np.linspace(-0.1 * np.pi, 2 * np.pi, 30)
y_1 = np.sinc(x) + 0.7
y_2 = np.tanh(x)
y_3 = np.exp(-np.sinc(x))

fig, ax = plt.subplots(1, 1)
ax.plot(x, y_1, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C0')

ax.plot(x, y_2, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C3')

ax.plot(x, y_3, color='k', linestyle=':', linewidth=1,
        marker='o', markersize=5,
        markeredgecolor='black', markerfacecolor='C2')

ax.legend(labels=["y_1", "y_2", "y_3"], ncol=3)

# inset_axes 方法
axins = inset_axes(ax, width="20%", height="20%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)
axins.plot(x, y_1, color='k', linestyle=':', linewidth=1,
           marker='o', markersize=5,
           markeredgecolor='black', markerfacecolor='C0')

axins.plot(x, y_2, color='k', linestyle=':', linewidth=1,
           marker='o', markersize=5,
           markeredgecolor='black', markerfacecolor='C3')

axins.plot(x, y_3, color='k', linestyle=':', linewidth=1,
           marker='o', markersize=5,
           markeredgecolor='black', markerfacecolor='C2')
# 调整子坐标系的显示范围
axins.set_xlim(0.6, 1.2)
axins.set_ylim(0.5, 1.0)
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

# # zoomed_inset_axes 方法
# axins = zoomed_inset_axes(ax, zoom=3, loc=4)
# axins.plot(x, y_1, color='k', linestyle=':', linewidth=1,
#             marker='o', markersize=5,
#             markeredgecolor='black', markerfacecolor='C0')
#
# axins.plot(x, y_2, color='k', linestyle=':', linewidth=1,
#             marker='o', markersize=5,
#             markeredgecolor='black', markerfacecolor='C3')
#
# axins.plot(x, y_3, color='k', linestyle=':', linewidth=1,
#             marker='o', markersize=5,
#             markeredgecolor='black', markerfacecolor='C2')
# x1, x2, y1, y2 = 0.8, 1.2, 0.8, 0.9
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# # fix the number of ticks on the inset axes
# axins.yaxis.get_major_locator().set_params(nbins=7)
# axins.xaxis.get_major_locator().set_params(nbins=7)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()
