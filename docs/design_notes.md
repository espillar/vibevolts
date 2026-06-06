# VibeVolts Design Notes

# Vibevolts per se
Creating a new version of volts that runs faster using jules and Gemini-cli
# Notes: infrastrucure, some code ideas
## Clocks: how to run discrete events
The fundamental idea for these deep sky threats is that the sensor will
be scanning much faster than the satellites move, so the clock rate is
set by the sensors readouts with the satellites frozen.
- Each sensor has next measurnebt time and a integration time.
- a master scheduler looks for the next cohort of sensors to
  be read out at the same time, sets the time there,
  runs the sensors and updates their clocks.
## Infrastructure
### Jules
Appears to hook into the system right,  including when logged in as 
earl.s

### GCE
Vibevolts

https://safe.menlosecurity.com/https://sites.google.com/afresearchlab.com/portal
(go ahead and give it your email if menlo asks)

https://safe.menlosecurity.com/https://console.cloud.google.com/vertex-ai/workbench/instances?project=afrl-il2-sbx-dnkc1-nfor

Instance *16047
 
git pull is set up - so start with that. Make sure you are up to date
gemini-cli is instaleld.  Done with a pip install gemini-cli, and then doing
a brief authentication dance.
`
## Numpy scatter gather
import numpy as np


data = np.arange(18).reshape(6, 3)
mask = np.array([True, False, True, False, True, False])
selected_rows = data[mask]
computed_results = selected_rows.sum(axis=1, keepdims=True)
final_array = np.hstack([data, results_container])

# Prompts
## LaTeX Documentation
Create a latex documentation file called vvdoc.tex. It should have the following sections:
Create a table of content which will be based off the section structure
you are generating below.
Proceed as follows:
- read in an individual python file
- create a new section
- write a summary of the purpose of file
- for each function in the file, write a summary of the purpose of that
  funtion and the arguments and return value.
- be absolutely sure that for any underscores in the file, escape them with
  a backslash BEFORE you write them out to the latex file
- do this for each of files
Write another section summarizing what the demos are. Again be sure
that the underscores in all text are preceeded by a backslash.
Next, for each of the files, create a new section and output
the contents of the file inside a new vebatim environment,
but before you write a line wrap the text if it is longer than
100 characters. In side the verbatim environment you do not need to
prefix a backslash to the underscore.

## prompt to create datastructure.{org, md}
There are a large number of functions that add to or modify
the dictionary sim_data.  Please go through all
   the files and create a document that creates a
   list of all the dictionary elements added by anybody to 
  sim_data, lists the types, and on the same lines
  lists what functions either add to or change that entry.  
  If a function creates a dictionary or item inside
  another, document it the same way.  Indent top level 
  entries more if they are subentries.  Put this data
  in a new file called datastructure.md, but do not 
  change any other files. Be sure to include information on the
  content and stucture of the detector array.

  
To this document add another section that lists all the functions
that display the data in hte current code base. Do not change any
other files.


In addition to datastructure.md, keep that file, but make a copy
in org format in datastrucure.org

