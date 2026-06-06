# VibeVolts Tasks

# Details
## Coding [12/26]
### Get a simple search pattern going [0/8]
- [ ] write some code that computes what been detected and do a 3-D map
- [ ] This should work for FOV intervals both long and short compared to the
  propagation time, as long as things don't move TOO fast.
- [ ] Implement the plan that was proposed by Gemini 
  My Recommendation:
   1. Modify nextIntegration (or your calling loop) to return a
      structured object that includes the current timestamp.
   2. Use Option 1 (CSV/Pandas) for initial development—the ability
      to immediately open a run in Excel to "sanity check" the SNR
      and signal values is invaluable for simulation work.
   3. If the files exceed 100MB, switch the export line to .to_parquet().


  Would you like me to implement a "Logger" or "Collector" class in a new file to automate
this?

### Future work [0/5]
- [ ] We are using he same filter for all the detectors: split into cohorts
- [ ] It would be nice to consolidate all those demos a little better
  put the demo code in the same files as what they are testing
  make them all rely on a standard initial "build", chaning
  the particulars of what they need each time.  Maybe, pros and cons.
- [ ] Think about how we deal with multiple constellations in the code.
- [ ] disentangle detectors from satellites - 
- [ ] resolve issues with having multiple constellations, potentially with
  different names etc.

### Metric Calculations [/]
- [ ] Calculate the gap times- basically for each column,
  compute the interobservation distances and histogram
  them.
### General Improvments [/]
- [ ] Fix angle of sensor from circular to square
- [ ] Think about how  to associate detectors with satellites  

