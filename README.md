# emgineer
> emg process tool

## Description
> "emgineer" will help you with processing EMG data.</br>
> This has lots of tools: EMG decomposition, plot spike trains and etc.

## Usage
> ```
> import emgineer
> ```
> ### EMG decomposition
> ```
> est = emgineer.EmgDecomposition(n_motor_unit)
> data = {ndarray shape of (n_samples, n_signals)}
> est.fit(data)
> st_valid, emg_mu_valid = est.transform(data)
> ```
> ### plot spike trains
> - blah blah bla

## Install
> ```
> python setup.py develop
> ```

## License
MIT