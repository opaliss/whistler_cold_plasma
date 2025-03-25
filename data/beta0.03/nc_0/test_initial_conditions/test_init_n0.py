import numpy as np
import matplotlib.pyplot as plt

class simulation:

    def __init__(self,wdir):

        # read parameters
        f = open("%s/info.bin" % wdir, "rb")
        topology = np.fromfile(f,count=3,dtype=np.float64)
        Lsim   =   np.fromfile(f,count=3,dtype=np.float64)
        Nsim   =   np.fromfile(f,count=3,dtype=np.float64)
        dt_wpe =   np.fromfile(f,count=1,dtype=np.float64)
        mime   =   np.fromfile(f,count=1,dtype=np.float64)
        wpewce =   np.fromfile(f,count=1,dtype=np.float64)
        vthe   =   np.fromfile(f,count=1,dtype=np.float64)
        f.close()

        print(' ------ Reference parameters  ------ '  )
        print("Path to the simulation data %s:" % wdir)
        print("Lx=%g, Ly=%g, Lz=%g" % (Lsim[0],Lsim[1],Lsim[2]))
        print("nx=%g, ny=%g, nz=%g" % (Nsim[0],Nsim[1],Nsim[2]))
        print("dx=%g, dy=%g, dz=%g" % (Lsim[0]/Nsim[0],Lsim[1]/Nsim[1],Lsim[2]/Nsim[2]))
        print("dt*wpe=%g" % dt_wpe)
        print("mi/me=%g" % mime)
        print("Reference wpe/wce (hot density) = %g" % wpewce)
        print("vthe/c=%g" % vthe)

        self.Lsim = Lsim
        self.Nsim = Nsim
        self.mime = mime
        self.wpewce  = wpewce
        self.dt_wpe = dt_wpe
        self.B0 = 1/wpewce

        self.wdir = wdir
        
    def get_data(self,q,tidx):

        step = tidx
        fn = '%s/data/%s_%i.gda' % (self.wdir,q,step)
        dat = np.fromfile(fn,dtype=np.float32)
        nx = int(self.Nsim[0])
        ny = int(self.Nsim[1])
        nz = int(self.Nsim[2])
        dat = np.reshape(dat, (nx,ny,nz),order='F')
        return dat

    def get_mean(self,q,tidx):
        
        dat = self.get_data(q,tidx)
        res = np.mean(dat)
        return res



tidx = 0
s = simulation("../")

Bz = s.get_mean('bz',tidx)
nec = s.get_mean('neC',tidx)
neh = s.get_mean('neH',tidx)
ni  = s.get_mean('ni',tidx)

pe_xx  = s.get_mean('peC-xx',tidx)
pe_yy  = s.get_mean('peC-yy',tidx)
pe_zz  = s.get_mean('peC-zz',tidx)

betaC_par = 2*pe_zz/Bz**2
betaC_per = (pe_xx+pe_yy)/Bz**2

pe_xx  = s.get_mean('peH-xx',tidx)
pe_yy  = s.get_mean('peH-yy',tidx)
pe_zz  = s.get_mean('peH-zz',tidx)

betaH_par = 2*pe_zz/Bz**2
betaH_per = (pe_xx+pe_yy)/Bz**2

pi_xx  = s.get_mean('pi-xx',tidx)
pi_yy  = s.get_mean('pi-yy',tidx)
pi_zz  = s.get_mean('pi-zz',tidx)

betai_par = 2*pi_zz/Bz**2
betai_per = (pi_xx+pi_yy)/Bz**2

print(' ------ Summary of Loaded initial condition (average values) ------ '  )

print('Bz0/B0 = %g' % (Bz/s.B0) )
print('nec/neH = %g' % (nec/neh) )
print('nec/ni  = %g' % (nec/ni) )
print('ni/neH  = %g' % (ni/neh) )
print('ni/(neC+neH) = %g' % (ni/(nec+neh)) )
print('wpe/wce with ion (total) density = %g' % (np.sqrt(ni)/Bz) )
print('wpe/wce with hot density = %g' % (np.sqrt(neh)/Bz) )

print(' ------ Parameters of the cold electrons (average values) ------ '  )

print('fraction of total density  = %g' % (nec/(nec+neh)) )
print('parallel beta              = %g' % betaC_par)
print('perp.    beta              = %g' % betaC_per)
print('T_per/T_par                = %g' % (betaC_per/betaC_par) )

print(' ------ Parameters of the hot electrons (average values) ------ '  )

print('fraction of total density  = %g' % (neh/(nec+neh)) )
print('parallel beta              = %g' % betaH_par)
print('perp.    beta              = %g' % betaH_per)
print('T_per/T_par                = %g' % (betaH_per/betaH_par) )

print(' ------ Parameters of the ions (average values) ------ '  )

('fraction of total density  = %g' % (ni/(nec+neh)) )
print('parallel beta              = %g' % betai_par)
print('perp.    beta              = %g' % betai_per)
print('T_per/T_par                = %g' % (betai_per/betai_par) )

print(' ------ Additional temperature ratios ------' )
print('TeH_par/TeC_par                = %g' % (betaH_par/betaC_par/neh*nec) )
print('Ti_par/TeC_par                = %g' % (betai_par/betaC_par/ni*nec) )
