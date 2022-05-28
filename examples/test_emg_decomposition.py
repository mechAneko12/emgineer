import emgineer
import numpy as np

est = emgineer.EmgDecomposition(n_motor_unit=40, random_state=0)

np.random.seed(0)
emg_raw = np.random.normal(size=(1000, 50))
est.fit(emg_raw)
st_valid, emg_mu_valid = est.transform(emg_raw)
print(st_valid.shape)
