import csiread

csidata = csiread.Intel('dat/csi-sample.dat')
csidata.read()
csidata.get_scaled_csi()

print(csidata.csi)