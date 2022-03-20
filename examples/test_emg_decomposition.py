import emgineer
import numpy as np

est = emgineer.EmeDecomposition(n_motor_unit=10, cashe='test')

emg_raw = np.random.normal(size=(1000, 20))
est.fit(emg_raw)
st_valid, emg_mu_valid = est.transform(emg_raw)
print(st_valid.shape)
