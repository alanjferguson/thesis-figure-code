import numpy as np

import matplotlib.pyplot as plt

import ajf_plts

plt.style.use(["./ajf_plts/base.mplstyle"])

fig, ax = plt.subplots(figsize=ajf_plts.single_wide_figsize)

L = 20.0

x = np.linspace(0.0, L, 1000)

depth = 1.0

xc = 0.5 * L

delta = 0.15
crack_height = delta * depth

lc = 1.5 * depth

I0 = 1.0
Ic = I0 * (depth - crack_height)**3.0 / depth**3.0

I_x = np.where(np.abs(x - xc) >= lc,
               I0,
               Ic + np.abs(x-xc) * (I0-Ic)/lc)

plt.plot(x, I_x)

#plt.show()

ax.axhline(I0, ls=':', lw=0.8, c='k', zorder=-10)
ax.axhline(Ic, ls=':', lw=0.8, c='k', zorder=-10)

ax.axvline(xc, ls=':', lw=0.8, c='k', zorder=-10)
ax.axvline(xc-lc, ls=':', lw=0.8, c='k', zorder=-10)
ax.axvline(xc+lc, ls=':', lw=0.8, c='k', zorder=-10)

for loc in [xc-lc, xc+lc]:
    ax.annotate(text="",
                xy=(xc, 0.34),
                xytext=(loc, 0.34),
                zorder=-200,
                arrowprops=dict(ec="k",
                                fc="k",
                                arrowstyle="<|-|>",
                                shrinkA=0,
                                shrinkB=0))
    ax.text(x=np.mean([xc, loc]),
            y=0.35,
            s="$l_c$",
            fontdict={"ha": 'center', "va": 'bottom'})

ax.set_xlabel('Position, $x$')
ax.set_ylabel(r'$I\,(x)$')

ax.set_xlim([0, L])
ax.set_ylim([0.0, 1.1])

plt.xticks([0.0, xc, L],
           [r"$0$",
            r"$x_c$",
            r"$L$"])
plt.yticks([0.0, Ic, I0],
           ["$0$",
            r"$I_c$",
            r"$I_0$"])
ax.tick_params(axis=u'both', which=u'both',length=0)

fig.tight_layout()

ajf_plts.save_fig(fig)
