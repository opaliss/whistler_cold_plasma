import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# read parameters
f = open("../info.bin", "rb")
topology = np.fromfile(f, count=3, dtype=np.float64)
Lsim = np.fromfile(f, count=3, dtype=np.float64)
Nsim = np.fromfile(f, count=3, dtype=np.float64)
dt = np.fromfile(f, count=1, dtype=np.float64)
mime = np.fromfile(f, count=1, dtype=np.float64)
wpewce = np.fromfile(f, count=1, dtype=np.float64)
vthe = np.fromfile(f, count=1, dtype=np.float64)
f.close()

nz = int(Nsim[2])
Lz = Lsim[2]
B0 = 1 / wpewce

# time range 

ti = 734  # steps between outputs
te = 429390  # final step
ta = np.arange(0, te + 1, ti)

nt = ta.size

# transverse components of B in field-aligned coordinates
bx = np.zeros((nt, nz))
by = np.zeros((nt, nz))
pzz_eH = np.zeros((nt, nz))
pzz_eC = np.zeros((nt, nz))
pyy_eH = np.zeros((nt, nz))
pyy_eC = np.zeros((nt, nz))
pxx_eH = np.zeros((nt, nz))
pxx_eC = np.zeros((nt, nz))

W = np.hanning(nz)

# read data & accumulate it in 2D arrays bx and by
# first index : time
# second index : coordinate
for j, t in enumerate(ta):
    fname = '../data/bx_%i.gda' % t
    print(fname)
    bx[j, :]= np.fromfile(fname, dtype=np.float32)
    fname = '../data/by_%i.gda' % t
    by[j, :]= np.fromfile(fname, dtype=np.float32)

    fname = '../data/peH-zz_%i.gda' % t
    pzz_eH[j, :] = np.fromfile(fname, dtype=np.float32)
    fname = '../data/peC-zz_%i.gda' % t
    pzz_eC[j, :] = np.fromfile(fname, dtype=np.float32)

    fname = '../data/peH-yy_%i.gda' % t
    pyy_eH[j, :] = np.fromfile(fname, dtype=np.float32)
    fname = '../data/peC-yy_%i.gda' % t
    pyy_eC[j, :] = np.fromfile(fname, dtype=np.float32)

    fname = '../data/peH-xx_%i.gda' % t
    pxx_eH[j, :] = np.fromfile(fname, dtype=np.float32)
    fname = '../data/peC-xx_%i.gda' % t
    pxx_eC[j, :] = np.fromfile(fname, dtype=np.float32)

# perform FFT over z
bxk = np.fft.fftshift(np.fft.fft(bx, axis=1, norm=None) / nz / B0, axes=1)
byk = np.fft.fftshift(np.fft.fft(by, axis=1, norm=None) / nz / B0, axes=1)

# spectrum |bx(t,k)|^2 + |by(t,k)|^2
S = np.abs(bxk) ** 2 + np.abs(byk) ** 2

# values of k
k = 2 * np.pi / Lz * np.arange(-nz // 2, nz // 2)

# time index
twce = ta * dt / wpewce

## ===============
## Plot mode amplitude vs time
## ===============
# mode amplitude
dB = np.sum(S, axis=1)

# fit in the interaval t1 to t2
t1 = 30
t2 = 90

idx = np.logical_and(twce >= t1, twce <= t2)

tfit = twce[idx]
dBfit = dB[idx]

p = np.polyfit(tfit, np.log(dBfit), 1)

print(' The growth rate is gamma/wce=%g' % p[0])

fit = np.exp(p[0] * tfit + p[1])

f, ax = plt.subplots(1, 1)

ax.plot(twce, dB, lw=2)
ax.plot(tfit, fit, '--')

ax.set_ylabel(r'${\delta}B^2/B_{0}^2$')
ax.set_xlabel(r'$t|\Omega_{ce}|$')

plt.savefig(fname="amplitude.pdf", bbox_inches='tight')

np.save("t_nc_0.npy", twce)
np.save("dB2_nc_0.npy", dB)

np.save("pzz_eH_nc_0.npy", np.mean(pzz_eH, axis=1))
np.save("pxx_eH_nc_0.npy", np.mean(pxx_eH, axis=1))
np.save("pyy_eH_nc_0.npy", np.mean(pyy_eH, axis=1))

np.save("pzz_eC_nc_0.npy", np.mean(pzz_eC, axis=1))
np.save("pxx_eC_nc_0.npy", np.mean(pxx_eC, axis=1))
np.save("pyy_eC_nc_0.npy", np.mean(pyy_eC, axis=1))

## ===============
## Plot spectrum
## ===============
fig, ax = plt.subplots(1, 1)

pos = ax.pcolormesh(twce, k, S.T, cmap="nipy_spectral", norm=matplotlib.colors.LogNorm(vmin=1e-10, vmax=1e-3),
                    shading='gouraud')
cbar = fig.colorbar(pos)

ax.set_ylim(0, 2)
ax.set_ylabel(r'$kd_e$')
ax.set_xlabel(r'$t|\Omega_{ce}|$')
ax.set_title(r'$\delta B^2/B_{0}^2$')

plt.savefig("spectrum_nc_n0_0.png", bbox_inches='tight', dpi=300)

## ===============
## Plot Bx
## ===============

f, ax = plt.subplots(1, 1)

im = ax.imshow(bx.T, origin='lower', extent=[0, twce[-1], 0, Lz], aspect='auto')

cb = plt.colorbar(im)

ax.set_ylabel(r'$z/d_e$')
ax.set_xlabel(r'$t\Omega_{ce}$')
ax.set_title(r'$B_x(t,z)$')

# plt.savefig(fname="bx.pdf", bbox_inches='tight')

plt.show()
