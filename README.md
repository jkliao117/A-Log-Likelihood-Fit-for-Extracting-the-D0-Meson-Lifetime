# A-Log-Likelihood-Fit-for-Extracting-the-D0-Meson-Lifetime
Computational coursework I completed for the Computational Physics modlue

## Abstract
The average lifetime of the D0 meson was determined by the minimisation of the nega- tive log-likelihood of the decay function from the experimental data containing 105 measure- ments of the decay time and error. The behaviours of the decay functions were explored and two minimisers were developed. The result achieved with the background signals included is 0.4097±0.0048ps, which is in good agreement with the value given by the Particle data group.

## Instructions
The code is written in Python 3.6 with the standard coding style.
Several packages including NumPy, Scipy, Matplotlib, math were imported.
The comments sometimes include equations that require referencing the lecture notes.

Files List:
part1.py		data reading, true decay function, true decay NLL
part2.py		parabolic minimisation, error evaluation 
part3.py		total measurement function, total decay NLL
part4.py		quasi-newton minimisation, error evaluation 
integration.py		trapezium rule integration
error_analysis.py	error for different sample sizes	
lifetime.txt		measurement data
README.txt

How to invoke the code:
The code should be able to run in any integrated Python development environments.
One recommended approach is:
1. Install Anaconda Navigator 1.6.2
2. Open the files with Spyder 3.3.1
3. Open an IPython console
4. Set the directory to be at this file
5. Run the code in the results.py file
And the outputs will be displayed in the console or pop up as figures.
