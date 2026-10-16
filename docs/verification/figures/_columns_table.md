| Case | Altitude | lmax | RMS of the field | columns minus dipole sum: max (nT) | rms (nT) | field above lmax: rms (nT) |
|---|---:|---:|---:|---:|---:|---:|
| cells | 100 km | 185 | 1.45 nT | 8.17e+00 | 1.58e-01 | 1.58e-01 |
| cells | 100 km | 899 | 1.45 nT | 7.07e-05 | 3.22e-07 |  |
| cells | 300 km | 185 | 0.208 nT | 1.02e-02 | 2.77e-04 | 2.77e-04 |
| cells | 300 km | 899 | 0.208 nT | 1.33e-13 | 8.75e-15 |  |
| cont_checker | 100 km | 185 | 22.4 nT | 8.88e+00 | 8.60e-01 | 8.60e-01 |
| cont_checker | 100 km | 600 | 22.4 nT | 8.47e-03 | 5.84e-04 |  |
| cont_checker | 300 km | 185 | 10.2 nT | 1.32e-02 | 1.69e-03 | 1.69e-03 |
| cont_checker | 300 km | 600 | 10.2 nT | 4.36e-11 | 3.90e-12 |  |
| cont_geology | 100 km | 185 | 11.4 nT | 6.12e+00 | 3.94e-01 | 3.94e-01 |
| cont_geology | 100 km | 600 | 11.4 nT | 7.28e-03 | 3.10e-04 |  |
| cont_geology | 300 km | 185 | 4.77 nT | 1.05e-02 | 7.93e-04 | 7.92e-04 |
| cont_geology | 300 km | 600 | 4.77 nT | 4.35e-11 | 1.95e-12 |  |

| Case | Coefficient check | Max relative difference |
|---|---|---:|
| cells | columns vs remit transform (max rel) | 4.2e-14 |
| cont_checker | float64 columns vs remit transform (max rel) | 3.0e-14 |
| cont_checker | float32 columns vs remit transform (max rel) | 1.7e-06 |
| cont_checker | float32 minus float64 columns, Br at 100 km (max, relative to the RMS field) | 6.6e-06 |
| cont_checker | float32 minus float64 columns, Br at 300 km (max, relative to the RMS field) | 1.9e-06 |
| cont_geology | float64 columns vs remit transform (max rel) | 4.7e-14 |
| cont_geology | float32 columns vs remit transform (max rel) | 3.1e-06 |
| cont_geology | float32 minus float64 columns, Br at 100 km (max, relative to the RMS field) | 1.1e-05 |
| cont_geology | float32 minus float64 columns, Br at 300 km (max, relative to the RMS field) | 1.7e-06 |
