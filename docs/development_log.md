# VibeVolts Development Log


# 2026-06
## [2026-06-06 Sat]
Well that was a bit of time off. I wasn't sure what I wanted to do,
what with some frustrations with getting the coding going, 
what with gemini changing to antigravity,  agentic programming coming
on stronger, and just ... life.

I've installed all hte antigravities, signed up for code tokens, learned 
a little about antigravity IDE, switched my work from emacs.  
I asked gemini to look at the code and it found a whole bunch of bugs.
In one way  this makes me feel bad, i another, it whiped out 10 bugs
I missed in no time flat.  Maybe I should have looked at the solutions 
more, but I think getting rid of some of that technical debit is probably a 
very good thing. 

I used mayb 60% of my daily credits, seems like that's not too much.  My eyes
feel dry.

# 2026-04

## [2026-04-19 Sun]
Almost a month has passed.  Clearly, not making daily progress here.
It seems


# 2026-03

## [2026-03-21 Sat]
Well, I was pretty busy the last couple weeks! I haven't made
the progress i wanted to, and now Ive forgotten where I'm at.

I know I was working on candencecontroller.  I need to review where
I am.  Also verify_cadence. Things seem to be working.

Tried to run pdoc to create documentation, but it's giving me troubles.
I don't know what's going on. Is this yack shaving? probably.  It
really should have worked though.

### Todos [1/8]
- [x] Get the VV docs in shape: having gemini work on that
- [ ] Implement data storage as recommended first a dictionary and then pandas.
  Do this by adding a python function dataHandling.
- [ ] perhaps a plot of point coverage
- [ ] gap time calculations and plots
- [ ] consider testing
- [ ] what of literate programming
- [ ] I also need to figure out why magit isn't pushing
  to the right target 


## [2026-03-08 Sun]
gemini-cli did quite a good job of vectorizing computations
in scandetectors, it all looked good but I may be missing something.
It understood the ins and outs of numpy and all the
options better than I do right now.

Gemini also had some informed thoughts about how to do the storage
### Todos [5/6]
- [x] review lambertian.py 
- [x] review the file cadencecontroller
- [x] Review the file scandetectors to make sure it's vectorized
- [x] run a test of the whole thing - alldemos.ipynb
- [x] Think about how I want to save data for processing
  write a short plan on changing the output of scandetectors
- [ ] Implement data storage as recommended first a dictionary and then pandas.
  Do this by adding a python function dataHandling.
- [ ] I also need to figure out why magit isn't pushing
  to the right target 


## [2026-03-07 Sat]


### Todo [2/3]
- [x] I need to check why afrl is not syncing right with github.
  OK, I switched to https and grabbed a token which will expire in june
- [x] go through and work with gemini-cli to implement the plan below.
- [ ] I also need to figure out why magit isn't pushing
  to the right target 

# 2026-02



## [2026-02-28 Sat]

Although I thought it through again, clock synchronization for
detectors is discussed in the architecture section.

The assumption right now is you have a 1:1 mapping between satellites
and detectors, each is an element of a vector of the objects. That's not
too bad; we could just duplicate satellites if we have more than
one sensor per satellite- a kludge but not too bad for a while! YAGNI

That leaves the question of how I should group sensors by integration time.
I will need that at some point, but is it better to do it now or a little
later?

An easy way to do it would be to establish a new structure that
includes a mask or list for the detectors that work at a particular interval,
that interval, and the time of the next readout.

I want to develop some code that will run scandetectors repeatedly,
but I might have several classes of detector with different integration times.
Therefor I want to establish some code that will create groups
of detectors with the same integration times, and iterate through
the groups when their integration is due to, then resetting the
next integration time based on the integration interval for that group.



I wold like the following names for the data structures:

Names:
- cadenceController.py has two functions
  - initCadence set up the variable 
  - nextIntegration - find and performs the next scan

The necessary data should be in a new     
- cadenceStructure - holds the data for cadenceController
  - nextTime, nextGroup the computed time for hte next integration, and the group it
    corresponds to
  - cadenceList (a numbered list) This contains cadence groups 
    - cadenceGroup (a structure) containing information on a group of sensors that
      have the same integration time 
      - scanInterval - time interval between new scans 
      - scanMask - a numpy mask that tells us which detectors have this interval 
      - scanNext - when the next integration is due, rest to the current time plus scanInterval
	everytime a scandetectors is called


Initialization:
- make a list of the integration times for the detectors (local)
- collapse that into a set. (local)
- create an empty cadenceList structure 
- take the for each item in the set (local)
  - create an cadenceGroup
  - create a mask for the relevant detectors -> scanMask
  - create a next integration time entry -> scanNext
  - create an interval entry - scanInterval
  - add to cadenceList

An iteration:
- set the global time to the next scanNext
- grab the appropriate mask for th
- propgate the satellites with that mask
- call scanDetectors with that mask
- for that particular key update the scanNext
- find the next next read time and put it in scanNext



### Todos [4/6]
- [x] Get on with planning the next phase
- [x] Check how detectors are bound to satellites and revise
- [x] figure out how to group detectors
- [x] Outline the clock schedular
- [ ] make nice variable names for the schedular
- [ ] start implementing the scanSchedular
- [ ] make sure your backgrounds match where you are space or ground - you
  should do this at work on paper to get a check
- [ ] Check the equation for required integration time w.r.t. flux



## [2026-02-24 Tue]

I essentially got photometry tucked in. I guess I want to do one more check.


### Todos [5/8]
- [x] While looking at the code found ['fixedpoints']['exclusion'] which is spurious?
  delete it. Code cleanup. Takes time though.
- [x] Look at the code we have and see how the tests should work
  looks like fixedTarget in radiometry_test, testdetector in detector.py
- [x] Get a couple of specific distances into notebook
- [x] vary the following variables and graph vs. expectations:
  size,   distance,   angles.  Those all check out.
- [x] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] Check the equation for required integration time w.r.t. flux
- [ ] make sure your backgrounds match where you are space or ground - you
  should do this at work on paper to get a check
- [ ] Get on with planning the next phase 


## [2026-02-22 Sun]
VV with potential interuptions.

### Todos [8/13]
- [x] add pixelOmege an and make sure its debuig printed
- [x] Check that our angular units pixelOmega work with the way
  we declare background flux - isn't
- [x] Get the units on pixelOmega to commensurate with our space and sky variables
- [x] Check that we are printing out the noise value and what the
  parameters are
- [x] do a pencil calculation of the noise value - this works
- [x] why is the size of the pixels less than 1 arcsec square?
- [x] is QE properly recorded here? 
- [x] Make the outputs look nicer
- [ ] Look at the code we have and see how the tests should work
- [ ] Get a couple of specific distances into notebook
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] Check the equation for required integration time w.r.t. flux
- [ ] make sure your backgrounds match where you are space or ground




## [2026-02-21 Sat]
Rebuilding my understanding of where I'm at. OK, that's not too bad.
Let's at it.
Have to get eglot working again.  I moved the pylsp server when I
killed the irritating anaconda and switched to conda.


### Todos [8/15]
- [x] Found that I wasn't including pixelOmega (which was an improved
  name I started using)
- [x] add pixelOmege an and make sure its debuig printed
- [x] Check that our angular units pixelOmega work with the way
  we declare background flux - isn't
- [x] Get the units on pixelOmega to commensurate with our space and sky variables
- [x] Check that we are printing out the noise value and what the
  parameters are
- [x] do a pencil calculation of the noise value - this works
- [x] why is the size of the pixels less than 1 arcsec square?
- [x] is QE properly recorded here? 
- [ ] Get a couple of specific distances into notebook
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] Check the equation for required integration time w.r.t. flux
- [ ] make sure your backgrounds match where you are space or ground




## [2026-02-16 Mon] Predsidents Day
A little distracted by PS not answering the phone and some afib,
staring a 0930.
Vectors were giving the wrong value for the lambertian function,
pi minus the angle.  I need a minus sigh.  I should look some more.

### Todos [7/14]
- [x] expose flux in satellite in code
- [x] expose flux on satellite by hand
  These agree once we notice one is in per cm2 and the other per m2
- [x] what's the value of lambertian function.
- [x] In the actual code test those points. If necessary
  refactor the code to expose them.
- [x]   Bisecting may  be a think here.
- [x] check what we're using for the integration time
- [x] refactor demo_fixedpoints.py because it was opaque 
- [ ] Check that we are printing out the noise value and what the
  parameters area
- [ ] do a pencil calculation of the noise value
- [ ] Get a couple of specific distances into notebook
- [ ] check that all the input paramters are what you think they are [0/1]
  - [ ] 
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] Check the equation for required integration time w.r.t. f


## [2026-02-15 Sun]
Not going well: I think my thoughts and actions have been
meandering. I need to be more specific and actionable.


### Todos [7/14]
- [x] Check alldemos to be sure you're all good to start.
- [x] Work out the rMPP the solar photon flux at a detector.
  Order of magnitude or so in two columns, including units
  in one column: it is all multiplicative.
- [x] Check a GEO magnitude is close
- [x] First try to match the whole chain to signal - massively off
- [x] Choose some good breakpoints to compare with actual
  code and get those partials worked out.
- [x] refactor so that lamberian is stand alone, check
  net photons incident on satellite
- [x] run the demo 'cause


- [ ] In the actual code test those points. If necessary
  refactor the code to expose them.
- [  ]   Bisecting may  be a think here.
- [ ] Compare with numbers from the Mathematica notebook
- [ ] Get a couple of specific distances into notebook
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] Check the equation for required integration time w.r.t. f
  
  

## [2026-02-14 Sat]
Trying to follow up on whwat gemini gave me but I'm kind
of lost on how to do it!  Wasted 20 min and came
back to reset. What do I want to do?  I want to follow the
values up the chain from the primitives and see if
they build up properly.

Installed the command line version of jules that I
didn't know existed.  Have gemini reorder.
Re-read. Reordering was very helpful but I don't yet see what's wrong.
Grrr.

OK. let's break it in the middle.  Print out the inputs to to
lambertain sphere and the outputs and see if I agree.




- [ ] Compare with numbers from the Mathematica notebook
- [ ] Get a couple of specific distances into notebook
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
-  [ ] Check the equation for required integration time w.r.t. 

  
## [2026-02-13 Fri]
I need to focus on debugging this chunk of radiometry code
so I can get on with "life."  I need to trace what's going on with the
signal calculation one line at a time.

I wonder if I can have GEMIMI help me with this- actually have it
start a process while I go ahead and do it the old fashion way.

Stopped after not enough time by AI class, which was interesting


### Prompts
The value calculated for signal in scandetectors.py seems incorrect to me.
Inside the code in this directory, please trace back all the variables
and algorithms that contribute to this value and produce a document
signal.md that records step by step how it is built up from it's constituents.
If there are single constants that contribute any any step include
them.  Reference what file any change is made in.

### Todos [1/10]
- [ ] Signal is totally broken. Do a line by line debug.
- [ ] make sure the details of this case agree with what you're testing in the code
- [ ] compare with a base case and fix
- [ ] put in equations document
- [ ] Compare with numbers from the Mathematica notebook
- [ ] Get a couple of specific distances into notebook
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?

## [2026-02-08 Sun]
Still most  things running from yesterday- much faster to start.
I think some of my workflow processes are kinda broken, or
at least not aligned well. Outlining what I'm doing really does
help.

Mess around with figuring out query-replace to get checkmarks right

I forgot that testObjects.py does most or all of what I need to do
to set up and test radiometry.  How should I fix this? I renamed it
radiometry_test.py


### Todos [16/25]
- [x] keep track of units the whole way in secondary lines
- [x] Choose V band
- [x] Get the solar brightness in magnitudes
- [x] Convert to photons/sec on a $1 m^2$ satellite
- [x] Assume face on, calculate the lambertian flux at earth or at that range
- [x] multiply by area of the telescope and any efficiency factors
- [x] Multiply by integration time
- [x] output this number
- [x] Find the background flux
- [x] convert to photons per second per meters per $\Omega$
- [x] multply by area, $\Omega$, efficiency, integration tiem
- [x] output this number
- [x] computer SNR by flux over $\sqrt{noise}$
- [x] output this number
- [x] convert the equations to LaTeX and compare with cognion:
  equations look good
- [x] Figure out what variable you need to set to do the right comparison.
  Does scandeteectors need to be refactored?
- [x] rename fixedObjects.py to radiometry_test.py
- [ ] Check worksheet results against MMa results.
- [ ] Signal is totally broken. Do a line by line debug.
- [ ] make sure the details of this case agree with what you're testing in the code
- [ ] compare with a base case and fix
- [ ] put in equations document
- [ ] Compare with numbers from the Mathematica notebook
- [ ] Get a couple of specific distances into notebook
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?

  
## [2026-02-07 Sat]

That twas an unfortunate week 1/2 occupied by Jury duty, most
of the rest by Furlough. 

Fixedtest.ipynb almost has the tests I need run.
Lets check what I have and go forward.

Slow progress.

### Todos [1/7]
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] fill in the elements in the notebook
- [x] convert the equations to LaTeX and compare with cognion:
  equations look good 
- [ ] Get a couple of specific distances into notebook
- [ ] compare with a base case and fix
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles





## [2026-02-03 Tue]
Last week was not a productive week. Furlough today helps.

NOTE working on the lambertian function in fixedtest notebook

### todo [4/10]
- [x] put geodemo back into the  demos
- [x] Record some of the debug information i learned in obsidian
  "python breakpoint debugging."
- [x] outline the computation  I need to do for radiometry checking
- [x] follow up to see how size is used in fixed target for the
  targets.
- [ ] follow up to see how the aperture size is expressed in all the
  code. What is it?
- [ ] fill in the elements in the notebook
- [ ] Get a couple of specific distances into notebook
- [ ] compare with a base case and fix
- [ ] Now we need to get all the scaling laws right
   configure the new fixedtest.ipynb to test radiometry-
- [ ] vary the following variables and graph vs. expectations:
  size,   distance,   angles



# 2026-01

## [2026-01-26 Mon]
Got bumbed out by an episode of afib yesterday :-(
Today trying to focus at work and get some stuff done
in the library.

Seems like I can't authenticate using my accounts on the Aerospace
network, but I can over my phone.  Might be able to use an API
key, but let's leave tht aside for today.

Argghh, some of those changes in signatues broke all_demos. Also
all demos is missing geodemo, where did that go? Maybe it's
only on Neptune?

I've been working for an hour and made almost no progress.
Break and thing about this.

Well, I learned another tool
import ipdb; ipdb.set_trace()


### todo [3/8]
- [x] Chage fixedSat so that new satellites are added when
  it is called again
- [x] get gemini to add docmentation to a few functions:
  scandetectors testObjects
- [x] import ipdb; ipdb.set_trace()
- [ ] configure the new fixedtest.ipynb to test radiometry-
- [ ] write asysocated tests in 
- [ ] Get a couple of specific distances in there 
- [ ] vary the following variables and graph vs. expectations:
  size
  distance
  angles
- [ ] computer the SNR "by hand" or from a paper and compare
  with the test case


## [2026-01-24 Sat]
Sitting down at 0830, I really want to get two hours in!

Anaconda was giving me problems and I wasn't able to update.
Clear it out, change to miniconda, from now on I will start with
conda activate dev.  

Return at 2045

### Prompts
Modify vvdocs.md and datastructure.md only. Preserve there basic
structure but update them to match the current state of the code.
Add a section to datastructure.md summarizing the contents of the
constants.py file.


In the file testObjects.py change the function fixedSat so that
if sim_data has an existing satellite, rather than overwriting the
existing satellite, the 'counts' 'satellites' counter is incremented
and the new satellite is added to the sim_data array of satellites

(There was some banter to get this to work)

In the function demoFixed in the file testObjects.py I would like to add
some analysis after the current figure.
In demo
to add which consists of the target being moved from the position
it's in a 100000000 along the x axis to 1/2, 2, 3, 4, and 5 times that distance.
The 

### todo [3/8]
- [x] figure out which md files are worth it and update them.
  Prepare to print them.
- [x] Once and for all remove anaconda and switch to miniconda
- [x] Install required packages in dev and run alldemos
- [ ] Chage fixedSat so that new satellites are added when
  it is called again
- [ ] configure the new fixedtest.ipynb to test radiometry-
- [ ] write assocated tests in 
- [ ] Get a couple of specific distances in there 
- [ ] vary the following variables and graph vs. expectations:
  size
  distance
  angles
- [ ] computer the SNR "by hand" or from a paper and compare
  with the test case

## [2026-01-23 Fri]
Sucky week in terms of getting work done here.
Got little done since I was prepping for AI
### todo [2/5]
- [x] Review last couple of days lessons
- [x] check that detector area is defined and derived right
- [ ] configure the new fixedtest.ipynb to test radiometry
- [ ] Get a couple of specific distances in there 
- [ ] vary the following variables and graph vs. expectations:
  size
  distance
  angles
- [ ] computer the SNR "by hand" or from a paper and compare
  with the test case

## [2026-01-19 Mon]

Making very slow progress today. Too many things moving around that
I'm not tracking.
Irritating I had emacs hanging, appeared to be because I
had a very large ipynb file that had a huge changeset.
Was I not running magit on these before?  

Trick with Jupyter kernel debugging: you need to create a
console and link it to the notebook you are running!


### todo [5/10]
- [x] run scandetectors demo to see what it's doing
- [x] update to the latest jupyter
- [x] do a conda update --all
- [x] try out debugging a little in jupyter notebooks
- [x] get debugging going with breakpoints so  you can see
  what's going on
- [ ] configure the new fixedtest.ipynb to test radiometry
- [ ] check that detector area is defined and derived right
- [ ] clean up scandetectors so it not printing so much
- [ ] vary the following variables and graph vs. expectations:
  size
  distance
  angles
- [ ] computer the SNR "by hand" or from a paper and compare
  with the test case

## [2026-01-18 Sun]
Travel interruptions to work, although I did have some good meetings
out there!

OK, the big thing to debug now is the SNR ratio for a satellite.
I think I need to set up something where I create one use case
to make work. You know.  (I wonder if literate programming would help??)

Interesting, I am doing most of my work today with prompts, and I am
getting somewhere. IF i had a relatively good overview of all the variables
I might be able to do it pretty quickly by hand.

Too a while but I did make some pretty good progress leveraging AI
quite heavily.  

### todo [4/5]
- [x] you may need to do some cleanup to do all of this.
- [x] create a test case with satellite looking straight up away from
  the sun with a satellite straight above it at a known range.
- [x] Set up so you can do the following
- [ ] vary the following variables and graph vs. expectations:
  size
  distance
  angles
- [x] computer the SNR "by hand" or from a paper and compare
  with the test case


### Prompts used

To the python file celesitalbodies.py add a function fixSun(sim_data) that creates a
  sun by setting the sun on the negative x axis at the normal distance to the sun by
  setting the variable sun_coords and transforming them to GCRS
  
In a new file test fixedSat.py create a function fixedSate(sim_data) that 
Creates a single satellite fixed at the surface of the earth along the positive x axis,
converted to GCRS, with 
with a single detector with a 2 pi field of view poinging up  so it can see everything.

In the same file create a function fixedTarget(sim_data,size,  x, y, z) that first converts
the x y z coordianates to GCRS and then places a fixedtarget at that position with size size.


add a new function called demoFixed to the testObjects.py file that runs the other
  functions in the file, and then displays the positions of the sun, the stellite, the
  test object, and the viewing vector in a 3d plot, for the distance from the earth,
  convert that radius length to the log of the radius distance so the points all fit on
  the same plot and separations from the earth remain visible.

  OK, something isn't working right here. In the testObjects.py file, remove all the
  conversions to GTRS coordinates, stay in the original x y z coordinates.  Rerun the
  plot.

  move the satellite to 1 meter along the x axis from the center of the earth and rerun
  the demo

change the fixedSat function so it takes x y and z coordiants and make sure you change
  the demoFixed function so that it feeds those variables.  make x 100 instead of the
  current 1 and rerun the demo.

 at th eend of demoFixed, run scandetector from scandetectors.py and print the output.
  Make sure demoFixed is the all_demos.py function.

  
## [2026-01-11 Sun]

Progress, I got through calculating SNR and a mask, but things are
not check out and working right yet.  Still, progress, at least
I have something to debug now. 

### Todo [6/6]
- [x] run through alldemos
- [x] test scandetector
- [x] what was I doing in patch? Merge it in somewhere
- [x] Try to run through scandetector
- [x] change variable names aper to aperArea to make more consistent
- [x] run through all demos one more time
  


## [2026-01-10 Sat]
Well, this was evidently not a good week!
First, lets move over our .emacs file from the jupiter.
Had to install magit "by hand" but the instructins in .emacs
were there! And helped!
Took about 22 minutes, not too bad actually. Sorted some laundry.
What's next?
Hmm, somehow flymake works now, and it's fixing my formatting. Wasted
some time on this, but I do think it's ultimately useful!
Looks like there were some more mistakes in the dimensions of arguments.
I think some may have been introduced last week. Letting gemini
clean things up.

### Todo [3/5]
- [x] port my emacs setup over.
- [x] figure out why lambertian is busted, or what's wrong
  with its arguments
- [x] run all demos
- [ ] start running the other guy and label better

  

@
## [2026-01-03 Sat]

### Todo [5/9]
- [x] Record some of the architectural ideas I had while walking
- [x] move fluxes into radiometry.calcs
- [x] move the size out of generate_log_spherical_points to targets
- [x] test that things are still running in demos
- [ ] Collect all fluxes of the marked objects
- [x] get the backgrounds
- [ ] calculate the SNRs
- [ ] capture the results in some sort of database
- [ ] start writing tests for SNRs

## [2026-01-02 Fri]
Check which targets are in view, calculate SNR,
tabulate those that are right, get ready for statistics,
do some checking.

### Today [4/7]
- [x] Reveiw what we have to do and write this list
- [x] create a function compareToVectorMask
- [x] check that lambertian sphere works adequantely for vectors of
  all the pieces
- [ ] check that lambertian is using the right "base brightness"
- [x] Compare the the angles to the acceptance angle and create a mask for those
- [ ] For those in the mask, computer the SNR.
- [ ]  store an appropriately labeled vector with the detector number, the target number, the
       time, and the SNR in a pandas array, or maybe a numpy array.



## [2026-01-01 Thu] New Years Day
New approach, build an interactive ad-hoc list below.
Success, got lots of messy variables moved to the right places.

### Today's ad-hoc checklist [10/10]
- [x] recreate todays list
- [x] fix the detectorVect in scandetectors
- [x] why is detectorVect not set up? See below
- [x] Remove any default creation of pointing_spheres
- [x] I need a function like "pointingInit"
- [x] Set up pointing spheres
- [x] follow the example in constellation to initialize detectors
- [x] how is it right in the demos?
- [x] Continue stepping through scandetectors, using jupyter for the moment,
  fixing a line at a time. Just check what you have is right.
- [x] Get to the stage where we are getting angles right.

### Tasks for AI [/]
- [ ] Respin documentation.




# 2025-12
## [2025-12-31 Wed]
Lost a couple of days there.  Maybe lost some comments too? Not sure.
### Today's ad-hoc checklist [9/21]
- [x] Check that the demos all run. They do.
- [x] Review where we are at and start adding items.
- [x] what is detectorVect and what should it be?
  Answer, it should be the pointing vector, but this has been placed
  in satellite, NOT in the detector structure.
- [x] find and move all initializations of pointings to somewhere in
  detector code - have gemini give you a report to help.
- [x] change update_satellite_pointing to update_detector_pointing
- [x] find out every place that is called and retarget
- [x] argg.  Since I've declared detector as a simplenamespace instead
  of a dictionary, references to it are different then references to dictionaries.
- [x] look for any spare ['pointing']
- [x] look for spare ['pointing_state'] 
- [ ] pointing state dimensions seem incompatible in pointing?
- [ ] move pointing_state and pointing from inside satellites to inside detector
- [ ] fix the pointing demo and get it to run.
- [ ] check where generate_poining_sphere puts stuff
- [ ] constellation.py
  - [ ] move references of pointing and pointing_state to detectors
- [ ] in propagation.py what are we even doing with the pointing?
- [ ] check on celestial_update
- [ ] check on propogate satellites
- [ ] build a links between detectors and satellites - start with 1:1, just
  and index, to extend
- [ ] fix the detectorVect in scandetectors
- [ ] Start stepping through scandetectors, using jupyter for the moment,
  fixing a line at a time. Just check what you have is right.
- [ ] Figure out how to move the pointing stuff into the detector code.
- [ ] outline what we need to do in comments in scandetectors.py
  
## [2025-12-28 Sun]
Getting a late start after worldly commitments. A little ADHD or maybe
the right word is yak shaving.
- I need to get detectors set up right.
Get things kinda going inthe morning, break, back, spend a little
time understanding einsum, then really waste some time getting ipython
installed as the python interpreter in emacs.  Sigh.

## [2025-12-27 Sat]
eglot workig.  back to work.  Slow start. Thinking is fuzzy,
I'm distracted and somehat grumpy. I need to document better,
and get a calmer background sound running.
Getting better after better sound. Moving detector up from inside
satellite using gemini-cli. That worked.
Now back into scandetectors.
## [2025-12-26 Fri] Boxing Day
First, let me see if I can figure out how to get
autocompletion working in a python directory.  I think this
would greatly accelerate my progres.
TRY TO Install pyright using npm
Install lsp-mode lsp-ui-mode global-company-mode in .emacs
end up going to eglot after struggles.
## [2025-12-25 Thu] Christmas Day
I think I'm going to look in to magit, since I seem to be spending a
fair a mount of time in git! Learned the bare minimum in 15 minutes,
C-x g is what I need to do to start it up.  Looks very nice actually.

Rename scansensors.py scandetectors.py

## [2025-12-24 Wed] Christmas Eve
Update the todo list, it was a mess with too man doubleed features.
Make some progress on iterating through the sensor target pairs.
Get in a couple of good pomodoros, wish I did more, but some progress!

Building scansensors.py
## [2025-12-23 Tue]
Man.  Another week. Not doing well here. I need to make some progress.
Add a new file literatevolts.org to start reasining more carefully. OK, I didn't have a
variable for simulation time but was using sim_start_time for that purpose. Use gemini in a
sandbox to search all files.
Arg.  This seems more complex than it should be. Let's pause and try running everything. Ok,
seems to be working.  Let's stand up and let Jay outside.
Grr, I am getting lost in all the variables and all the odd ways they are geting called. rgrep
helps, but ... it's kind of painful. Let's go through and replace all references of
propagate_satellites_new with propagate_satellites, since that was confusing.
Lunch break and nap. 1334 Finish that. Let me spend time looking at all time references.

start_time will be the tag used in the data structure
time will be the tag for the current execution time stored in the strucutre
delta_time is the "auto-increment time" used by autoincrepment (somewhere).


## [2025-12-18 Thu]
set up on il4 GCE. 
## [2025-12-13 Sat]
Been a slow two weeks.  I need to do some focus here.

## [2025-12-08 Mon]
Refactoring a little, putting celestialbody code into celestialbodies.py instead
of elsewhere. Trying to see if gemini can handle this right.

## [2025-12-07 Sun]
Well that was a week without progress.
Didn't make TOO much progress today, I had a cold. However,
I did have gemini produce a nice documentation file
simulation_data_trace.md
that I think will be very helpful going forward.

# 2025-11
## [2025-11-30 Sun]
Having Gemini run the demos crashes immediately. On one hand this is irritating,
on the other hand it's a good check! Let's examine this deliberately.
It looks like some constants changed at some point: I think I'm going to let it change
things since it will be hard to trace all the changes.  I need to pay more attention though.
I wonder exactly who changed those variable names when? Somewhat frustrating.
Cleaned up my documentation on where I was going
Looks like some of the MD documentation is in correct, some functions are not where they
say they are. Task gemini to fix.
I can tell I'm not focusing, I'm being distracted by my tools and not in flow.
Back from lunch run the demos one time to be sure things are still basically running.
OK,figured the bug out by printing where things weren't agreeing on the same side and
feeding to gemini. Problem: copying references instead of values.
OK, calling that part done for now.
Move add_fixed_points to target.py
I think I will add a new module visibility to create and collect the visibilty data.
Or maybe I could put it in pointing?

## [2025-11-28 Fri]
All tests seem to be running.
I need to look at whether exclusions are making it in to the tracking part.
Got way to distracted by life
## [2025-11-27 Thu] Thanksgiving
Looking at the pointing functions. I think I underetand
Doing some demos to see if that's all looking good- generating a longer chain with one
satellite (gemini doing the grunt work.)
Task gemini with creating a greedy vector sorter to get the fibonacci sequence to be more
efficient, and have it write a test.
That seems to work well!  Add to all demos.
Rename visibility to exclusion for clarity.
Attempting to move exclusion into the pointing.py funcitons.
## [2025-11-26 Wed]
Fiddled around a little to get cloud shell working so emacs would work.
There are still some weird keypmappng problems but I'm working it.
Where was I? I ran the basic test and it worked... but now it's 2:30
and local 59 minute rule is in effect.  Well, I think I will scoot. 
## [2025-11-23 Sun]
Well this has been an awful week for progress it seems! Although I
have done some equation stuff in the TeX document and regarding FAXT.
Have gemini update the detector documentation since it was out of date.
Interesting that it knows about pandoc! Hmm, i keep hitting the escape key
and deleting what I'm doing in gemini.
OK, there were some bugs in the pointing demo that I got gemini to fix.
I did get something revised, cleaned up exclusion and got rid of some dross,
made some plots better, udpated the documentaiton and todos,
cleaing up the pointing.py file to make it clearier.
## [2025-11-15 Sat]
Challenge: what's the best way to switch things over to SimpleNamespace?
First lets try a few of the things we would like to see happen.
OK, try except or hasattr can both be used to check what's going on
in a SimpleNameSpace. Ask gemini to refactor detector.py first. It does that
and then goes ahead and does all the files and even runs the demos- and
they all passed.  Amazing. OK, let's move filt into detector.
## [2025-11-14 Fri]
Tryihg to get things done in the office. 
Realizing that I don't have a good understanding of my initializatoin. Create
a new function sim_check that helps by running through the data structure and tells
me when things don't exist. I think his will be useful. Lets check this in.

Thinking about how to do the hierarchy of functions underneath a satellite.
Underneath a satellite lets have a pointer to a  structure that's a detector.
Actualy, I think it might be best to make detector a class, since I may want to 
mutate varioius parts, and the alternative: a namedtuple- is immutable.
This is a bit more than I want to tackle on my small screen at work, and 
I am out of hours.
Later investigation convinces me I want to use SimpleNamespace instead.
## [2025-11-13 Thu]
Back at work in the office.  seeing if I can work this in 
a terminal window in GCE.  Seems sluggish, but maybe OK.I 
## [2025-11-12 Wed]
Continuing on the quest to get search working.  Took a while to get started.
Made a little progress. fibonacciSearch now holds the fibonacci code, so
I can have a different module if I want to search differently.
I need to run the tests again, even though I don't think I've changed much.
Failure of course. I'm doing too much internal restucturing. I need to
think about the way forward. Putting everyhing in this global vector
thing SOUNDS efficient, but I think it's premature optimization and may
be making it harder to understand all this code :-(

Later, want to get the demos working again. Everything working
except GeoConstellation- let's see if I'm making those
## [2025-11-11 Tue]
The search code is harder than I thought to think about.  I wrote some
things down on the remarkable, and then wrote down specs in the
search design section.
Creating fibonacciSearch.py to hold the new items.
## [2025-11-10 Mon]
Revelation about how to do search.
- Choose an interval for propagation short compared
  to the time it takes an object to move significantly across fields.
- Develop an algorithm to choose subsequent FOVs and remember how long
  they take.
- Every propagation tick, decide how many FOV's would have been used.
  Include the next FOVs in what was surveiled during the interval.
  Surveil those zones.
- Rollover fractions accumulate to the next time interval.
- This should work for FOV intervals both long and short compared to the
  propagation time, as long as things don't move TOO fast.

Maybe I should treat a detector with it's pointing as a structure with
some functions on it:
- creation builds the data structure including.
  Includes integration time.
- nextfield yields the next pointing vector (depends on what type of
  pointer) and the infield funciton. There may be a bunch of these-
  or maybe it's just a structure that's an argument to be used.
  Handles avoidance too.  (Will depend on
  satellite state).  Probably stores a history.

Let's build this out.
## [2025-11-09 Sun]
Went through and finished the comparison of the Mathematica numbers, the
ad hoc numbers based on 866000 photons per second per square cm,
and the code from the detector.py.
They agree, although the defition of what the photometry aperture is vs. a pixel
etc. needs to be payed attention to!

Well, it seems like for fairly reasonable apertures you might only need to integrate a few
seconds,  which means you might have to snapshot what's in the FOV every - I don't know,
second, which is a bunch. So what's the answer?

Maybe propagate the satellites every minute or whatever, but find the radiometry vectors
more frequently and search all of them for matches. So now you want the integration times to
be considerably shorter than the propagation time so we don't miss things.
I guess i need a check for this in the code.
Hmm, for smallish apertures and large photometry aperture the required integration time
could be 10 minutes. Maybe we need to set the step time to 10 minutes, which would
give you peculiar velocity tracks that are 9000 arcsec long: so that's unrealistic.  
### Pointing Arrays
#### 

## [2025-11-08 Sat]
More working in the notebook to understand why my photometry is off - or not.
## [2025-11-07 Fri]
I'm going to mess around with taking some note and do some calculations in a Jupyter notebook
called VVnotes.ipynb That works out to agreeing with the old MMa calculation pretty well,
although I was surprised that the MMa calculation took a photometry radius instead
of a photometry diameter.
I guess I need to look at my python code now!
## [2025-11-06 Thu]
OK, missed a day, but I did read a bunch of stuff on
AI for Air Dominance.
First I want to make sure my limiting magnitudes are about right.  I need to compare with my V
magnitude graph.
Well, that didn't work right.  Somehow the computation is not working.
Well, one mistake is I didn't put the right background in.
OK, so how to check this out. What artifacts are there?
- the python code
- the tex document
- the mathematica document

What's the fastest thing to compare?  Well my units were spread across three files
and were messed up.  Revise equations file and trace things down the chain.
Made a bunch of progress, I think, but things are still off.  Going to need to work
on this. Sigh.
## [2025-11-04 Tue]
OK, I think I  need to seperate out the detector stuff into it's own module so I can test it
right and see it in one piece!  Making progress on that
## [2025-11-03 Mon]
Working on getting all_demos to run, focusing right now on getting
makeDetector in constellation.py to run, just importing things in Untitled.ipynb
at the end. Probably go there and see what's going on. I should make the size of the  array not
a constant 9 or 11, but  constant defined in constants.py and propagate that. I'm having gemini
fix that, and also truncate the DETECTOR_ from all of the constants related to detector- just
too long.

I see what's wrong with my code. I was trying to put the photometry band, a string, into a
numpy array.  Need to fix that.
Fixed.  I got demo_requiredIntegrationTime working to graph what I wanted. I need
to check that the values are good still!
## [2025-11-02 Sun]
Tried to run the top level demo and it failed.
Ran gemini against the changes I may and it found a bunch of errors: let it fix
them. And then I messed up and had to redo it.

Messed with it for a hour. Some progress but boy am I dumb: going too slowly.
Not understanding the whole structure.  

# 2025-10
## [2025-10-31 Fri]
Get started on work, but Max Yates texts, and I need to update homebrew, so
that's thinking.
Loading up the dichromacy theme, its a little higher contrast.  Taking
forever to build Homebrew.  
Delete and load directly using NPM.
I think I have the limiting magnitude coded, but I need to review that!

## [2025-10-30 Thu]

Add a document Equations for etendue.tex to keep track of that stuff. I think this is
progress.


## [2025-10-28 Tue]
Did a little work on getting a limiting magnitude function.  I did up the
calculation in a new latex file equations for etendue.tex.

## [2025-10-26 Sun]
OK, I was sieck for some time with esophogitis. Let's get back to it.

- [x] Let's run of the old tests.
- [ ] Review the documentation we have and improve it a little by adding
  the plotting functions to the documentation

## [2025-10-09 Thu]
First figured out git rebase to clean up history a little, interactive.
Pretty neat, but somthing I will need to review!

Lets go through files and try to organize what we have. Well, that may
not have been too useful but I definately cleaned things up a little.


OK, let's look at and document how we build out a simulation.
I realize I haven't really got this caged in my mind and it leads to inability
to act, since I don't even know where to start.
Well I was kinda lazy, but I did generate a file
>>>
>  datastructure.{org, md}
that seems preety doamn useful!




### Todo  [3/5]
- [ ] Document the damn architecture in terms of how you build and run
  the simulation
- [ ] Check we are initializing the detector arrays and make sure you are filling
  all the fields.
- [ ] write a function in radiometry to compute it for a particular system
- [ ] Start building the functions to track what's seen - I probably need
  an outline for this.
- [ ] Check that vstack etc. are being used consistently when adding a new constellation.
  I think I'm doing this wrong.  




## [2025-10-08 Wed]
Update gemini-cli  to 0.8.1.  I thoght I did that yesterday, but I could have
been mistaken.

Wow, I didn't write the physics and interpretation of dwell time down right AT ALL.
Wriging it down from the book 
$$t = {\omega\beta\gamma^2 \over   \eta f^2 \alpha^2  A  }  $$

### Todo  [3/5]
- [x] Plot how bright something is as a function of distance. Choose a size, a band
  choose a solar angle, plot vs. distance out to lunar radii.
- [x] write down what I need get a search rate for a sensor
- [x] modify the sensor spec so it has those variables.
- [ ] Check we are initializing the detector arrays and make sure you are filling
  all the fields.
- [ ] write a function in radiometry to compute it for a particular system
- [ ] Start building the functions to track what's seen - I probably need
  an outline for this.
- [ ] Check that vstack etc. are being used consistently when adding a new constellation.
  I think I'm doing this wrong.  

## [2025-10-07 Tue]
Furloughed.  I did make a little progress yesterday, well, not so much on this!
Fixed less fairly quickly.
Got stuck up in doing things imenu- actually that was pretty easy, start getting
tangled up - should I say yak-shaving - on setting up LSP mode.  i mean that's nice,
but it's YAK SHAVING. STOP IT.

OK, now looking at creating a new simpler lambertian sphere function.
OK, got that really simple plot going. Interesting hubrid work, I DID use
gemini, but I did quite a bit by hand and used colab to solve a simple subproblem
without a lot of ceremony.  


### Todo  [5/9]
- [x] Get the damn vvdoc.tex file working in the background.  I might give this task
  to jules.google.com if gemini is taken up with it.
- [x] Find and fix a bug in the less.tex document
- [x] Brew update underway, a new gemini-cli. Brew upgrade failed.
- [x] Install gemini-cli following following gemini instructions using NPM.
  Fast and smooth
- [x] Plot how bright something is as a function of distance. Choose a size, a band
  choose a solar angle, plot vs. distance out to lunar radii.
- [ ] write down what I need get a search rate for a sensor
- [ ] modify the sensor spec so it has those variables
- [ ] write a function in radiometry to compute it for a particular system
- [ ] Start building the functions to track what's seen - I probably need
  an outline for this.
  

## [2025-10-07 Tue]
Furloughed.  I did make a little progress yesterday, well, not so much on this!
Fixed less fairly quickly.
Got stuck up in doing things imenu- actually that was pretty easy, start getting
tangled up - should I say yak-shaving - on setting up LSP mode.  i mean that's nice,
but it's YAK SHAVING. STOP IT.

OK, now looking at creating a new simpler lambertian sphere function.
OK, got that really simple plot going. Interesting hubrid work, I DID use
gemini, but I did quite a bit by hand and used colab to solve a simple subproblem
without a lot of ceremony.  


### Todo  [5/9]
- [x] Get the damn vvdoc.tex file working in the background.  I might give this task
  to jules.google.com if gemini is taken up with it.
- [x] Find and fix a bug in the less.tex document
- [x] Brew update underway, a new gemini-cli. Brew upgrade failed.
- [x] Install gemini-cli following following gemini instructions using NPM.
  Fast and smooth
- [x] Plot how bright something is as a function of distance. Choose a size, a band
  choose a solar angle, plot vs. distance out to lunar radii.
- [ ] write down what I need get a search rate for a sensor
- [ ] modify the sensor spec so it has those variables
- [ ] write a function in radiometry to compute it for a particular system
- [ ] Start building the functions to track what's seen - I probably need
  an outline for this.


## [2025-10-05 Sun]
Took me a while to get going, but I wrote down some prioriities on the remarkable and
they seem to be sticking.

OK, this is a little frustrating.  Gemini is NOT able to get report.tex formatted right.
I'm going to make sure everything is saved here and do a push to set jules loose on the
problem in the background. After a WHOLE BUNCH of dicking around that worked. Quite
frustrating.  Took way to long.

### Todo  [1/5]
- [x] Get the damn vvdoc.tex file working in the background.  I might give this task
  to jules.google.com if gemini is taken up with it.
- [ ] write down what I need get a search rate for a sensor
- [ ] modify the sensor spec so it has those variables
- [ ] write a function in radiometry to compute it for a particular system
- [ ] Start building the functions to track what's seen - I probably need
  an outline for this.

## [2025-10-04 Sat]
Well this has been a lost week! I need to see where I'm at.

Well, I need to get things searching at the right rate.
I think that depends on my adding a dwell time.
Well, didn't get too much done, but I did get some minimal work on documentation.
For whatever reason darn gemini is having a really hard time just fixing all the
underscores to backslash underscore in report.tex that it's generating.

Also irritated by some emacs things, like not seeing files with spaces
in them. checking out aquaemacs again.


# 2025--09

## [2025-09-28 Sun]
It's been a week with not as much done as I wanted.
Hmm, gemini says there's an update so I brew update, which is taking a while- looks
like node wants to do a bunch of stuff.
rename the old gemini-cli to gemini and updated node.  That all took 16 minutes.  Grrrr.
Well, I hope I'm set up now.

Cogitation. I want the simplest thing that will work. So create a single grid for 20 deg FOV

Constellation.py needs to manage the creation of a constellation, which includes
generating the satellites and their pointing characterisics.

Making some reasonable progress- well maybe not. I seem to be stuck on not getting
the right argument to geos, which is confusing. Maybe I should start spyder.

- lunch

Dumped everything from the system over lunch now everything works now.  Very odd.
I wonder what was sticking? Add a new test that plots the RA and DEC vs time as we march
through the space.  That works.  Somehow the plotting got messed up and wasted 20min
making plots come out right.

Interesting.  I think the word results or result was being overloaded somehow.

## [2025-09-21 Sun]
OK logging in and fixing up a glitch in the GEMINI.md documentation "automatgically"
after syncing things in.  My set up was still going from yesterday.

Improving documentation, it seems like some type hints are being added.  Good!

OK, what is next.  I guess the next thing in my list above was creating a good builder
function for the constellations.  Spent a little time reading up on the propagation.py
module, documenting it. 

Vibe coded something to create a constellation, at least put the GEOs in place.

Later lets do some work on making a scan pattern thingy. Actually first,
I think I want to make a global list of constellations to propogate.  Actually
let's save that for a little later.

Vibe code something to change the positions and a demo, but the pointing vector isnt
getting updatd. I need to work on this next.  I think it's time for a shower.


## [2025-09-20 Sat]
It's been a bad couple of weeks in terms of making progress.  I really need to review
the code I have and then move forward. Working on Neptune, syncing stuff down to
get things up to date.
-  Remove my brew installed verion of gemini and put in the npm version.
  It wasn't updating right, just didn't trust it.
- OK, got fed up with trying to get gemini-cli reinstalled, got into yak shaving down to
  getting node reinstalled with brew which was bucking me.  Bathroom and mess around.
- Watch some ER
-  Reinstall node from nodes.org.   Install gemini-cli.  Things are working now, back to where
   I was!

- Later in the evening actually install and things all up on golden adorable.  Try to get
  strong tex file up, but that didn't work.  I need to try that again.
  

## [2025-09-15 Mon]
No gemini-cli installed on GCE by default. 

Well, I'm looking at things, but I'm not clear on the code,
so I'm spending too much time thrashing.


## [2025-09-14 Sun]
Long week without getting enough done. Looks like the first thing to try is
the demo, which runs "in the office" (on GCE?) but not at home?

OK, now that seems to work.  Who knows, maybe I had a corrupt python session.
So what's next?

Maybe a distraction, but let's try to install google-cli.  Homebrew.

OK, I think it's time to make the master structure an  empty dictionary to start, and then
have the various functions populate their pieces and add to the global.
Ask gemini-cli to do this. This is the kind of refactoring that's irritating
and problematic.  I'ld like to see if gemini can do this!

## [2025-09-08 Mon]
Try to run the same demo as yesterday in the office and here it
runs fine.  Confusing.  I didn't take notes that were very good.

OK, trying to figure out what's useful to do here. i think I'll generate a nice
LaTeX document  I can print out to review tomorrow using Jules. 

## [2025-09-07 Sun]
Getting started in the afternoon. OK. Got distracted by tax reviews in the morning.

Looked up scatter gather above, I think this will make calculations a lot faster.

I think what I should do is work towards scoring some simple constellations, and
get those done "by hand."  So let's write an algorithm and generate some prompts.
Write those above in details.

### P1
Follow instructions in agents.org.
 
=======
Create a new module called pointing vectors. In it
create a new function called pointing_vectors(n) which 
creates n equally spaced
points on the sphere using the fibonacci sphere algorhtm, and return the
3 dimensional points in an n by 3 numpy array.
Also add a function to the module which draws the points 
on a 3-d sphere using
plotly.
Add a demo funciton to the demos that calls this function and 
displays it.

### P2
Follow  the instructions in agents.org.
For each satellite, add two more parameters:  first a "pointing count" which
indicates what spherical pointing grid will be used.  Second "pointing place"
which indicates where in the sequence of pointing we are currently at.
Create a function "pointing place update" that increments the pointing place
value for all of the satellites, and if the pointing place is greater than or
equal to pointing count resets pointing place to 0.

Create another dictionary in the simulations state dictionary called
pointing spheres.  Entries will be indexed by the number of points the
sphere is divided into, and the content will be numpy arrays created by
the pointingVectors function.

In the simulations state dicionary add another variable called delta_time that
refers to the time between time steps in the simulation.
 



## [2025-09-06 Sat]
After reading an article and seeing how things were going, fed Jules the following prompt:


### Prompt1

Without changing any of the function signatures or functionality,
refactor the python files so that there are ideally no more than 150 lines per file.
Update the .org and .md documentation files so they are current. Before you
proceed give me a plan telling me what functions go in what files.I don't know
how this is going to work: but I'm going to try it out!

Took 18 minutes.  I need to review.

Worked well!  Spend some more time thinking about how to optimize search.

## [2025-09-04 Thu]
Tried this one in Gemini but it didn't go well: will have to try again. One problem
was that I was dragging things by hand, and even though there weren't
many externals left out gemini created plugs to replace them.  My bad.
I need to be careful about context.


Consider the following two files. I would like to create a new testing function that
creates a single satellite in a GEO location and initialized celestial bodies.
Create a two diminsional grid with the first axis going from -pi/2 to pi/2 in 50 in increments
correspnding to the latitude or declination, and the second axis going from 0 to 2 pi
correseponding
to longitude or right ascention in 100 steps. For each of these points, orient the pointing
vector
of the satellite in this direction and call the exclusion function. Place the return value in
the array.
Finally plot this array using plotly. Place this function in a new module separate from the one
I am showing you.


## [2025-09-01 Wed]  Actuall did some good thinking and notes on
things to do.  Started implementing thu.

## [2025-09-02 Tue]
Made a little progress getting things working in the cloud on the
work computer.  Things needed to be moved.

## [2025-09-01 Mon] Labor day.
I had left a thing running last night in jules, but it didn't finish
right- it could be that things timed out so things didn't get synced.
Anyway, asking it to fix the bug with graphics not being called right
again.  Prompt 2 from yesterday wasn't running.

I think I've been running too fast. Time to look at the code a little.

OK. Waste some time here. git mv rename this file. Figure out ^tab gets you
between tabs in safari.  Accidentally kill the Jules job I had fixing
the error listed above. Figure out how to close a "window" in emacs.
I guess this will all be useful but I wish I hadn't killed the job!

OK, it churns for some time- but it more or less turns out right this
time at last.  I need to edit some thigns.

Change which functions are called. Create a new funcTion just
creates the demo_plots file but isn't called by default (gemini).

Yeah, I think I really need to pay attention on how to get some
results more interactively after each change to see that things
are working - probably doing this in a jupyter notebook interactively
makes sense.  This is actually the kind of thing an AI should be able
to do pretty well.



### prompt 3



### prompt 2
add a function dump_detector that prints out a table with the rows
representing all the detectors and the labeled columns being the different
aspects of that detector.




### prompt 1
- Write function called jerk(satellite_number) that takes takes the satellite
  indicated by that number and moves the pointing vector 0.3 radians in
  any direction.
- write  function that examines the exclusion_table, finds any satellite
  (column) which entirely 0, and applies the function jerk to that satellite.
- create a new demo function that initializes the simulation,
  then creates and plots an exclusion_table, then applies the function
  just mentioned, and then creates and plots a new exclusion_table. 



# 2025-08


## [2025-08-31 Sun]
OK, lets get serious.


### prompt 2 - ran overnight and didn't work
- take all the demos that create plots in plotly, wrap them up in a new
  function that you run when you are testing things on the jules server
  and place the outputs in a new html file placed in the repo that
  can be viewed stand alone in the browswer.
- neglected to tell it to leave the original demos there- got that in there
  on the prompt.

### Prompt 1 - this is sort of a repeat  worked after I changed it to not do so much testing.
-read in and follow the instructions in agents.org file.
-Modify the number of fixed points created is only 100 for the moment.
-report any time you violate these instructions.
-Create another array visibility in overall structures that will have a number of columns equal
to the number of satellites and a number of rows equal to the number of fixed points. It will
be extended into a third dimension as the simulation proceeds.
-change the function exclusion so that it returns a 0 if pointing is exclueded and 1 if it is
not excluded.
-check that the function create_exclusion_table works with this new array and fills its
elements by calling the fuction exclusion.
-do not run all the demos, give me a change for GitHub.


## [2025-08-30 Sat]
Well, that was a couple of weeks with not much going on.  Got to stop that.

- An interesting idea that Hayden came up was that I might actually run some diagnostics
  in the virtual jules VM (or wherever, I guess!) and post them back to git.  I kinda like
that.
- i need to review status

Futzing around trying to get jules to autogenerate some good graphs of the code.
Clearly this is some sort of yak shaving

## [2025-08-23 Sat]
Hmm.  Too bad I left some dead time here.

- Have vibevolts update all the documentation.

## [2025-08-18 Mon]
Doing some coding in the hotel room in Kingman while Deborah gets ready to leave.
Hmm.
Well, that's kinda working, but somehow I am having some challenges getting git
to the way I want it too. Ther are some edge cases i guess.

Later work a little when I get home, still maybe some problems.

OK, well, later, clone it on to neptune, which isn't too demanding intellectually,
but a good thing to do if I'm going to work this in the long run.
Establish a nice ssh key for push and pull in git and on the local machine
and in the repo.  Git copilot helped with that!

Well I think I'm getting the hang of it, but I really ought to write it down.
For now, what's the next useful step I can take?

OK, I think I did something to do some visibility calculations. I haven't really
RUN it though to check if things are working. Next.

## [2025-08-17 Sun]

OK, I need to collect observations now.  Let's get a prompt.  Maybe see if jules
can do this since it's across several files now.

## Prompt
use the python tools currently in the repository, but don't change them
un-necessarily.
Create a new central data structure in vibevolts.py
called fixedpoints that is
initialized using the generate_log_spherical_points including
points from 2000000 meters to tice geodistances.  Add
a new demo functino that plots this data in a plotly.

Did some reading on git- I thought it was all in my head, put creating
local branches of remote things, switching branches, restoring older
versions of files, and newer commans switch and resotre were not
in my vocabulary.

## [2025-08-16 Sat]
Last Socorro Vacation Day. Testing out working copy. Seems really good
for some things! Took me 7 minutes to get my environment up.

OK, I need to take in to account points of view that are blocked by
earth or blinded by the sun moon or earth.  It would be nice
to make this an ECS function- but let's start simple

### Prompt - this appears to have mostly worked. a

Based on the existing code you've just read, create a new
python function exclusion  in a new file that does the following.

Add two global variables, earth_radius and moon_radius that contain
those radii in meters.
Create for me a function that takes an index number into the satellites
array, and extracts position for the satellite, the pointing
vector for the satellite, and also collects positions of the sun
and the moon.
Compute the unit vectors to the sun, the moon, and the earth from
satellite position.

For the sun, compute the angle betwen the vector to the sun and
the pointing vector, and set a flag if the angle is less than
the solar exclusion angle.

For the moon and the earth, calculate the angle between the
vector to the objects and the pointing angle, subtract
the arctangent of the  radius of the object and the distance to
to the object, and set flags if either is less than the
appropriate exclusion angle.

Set a global exclusion flag if any of these three flags is
set and return this flag, either true of false.


For testing, create a function that that does some displays in
plotly.  The function should initialize the positions of the
sun and moon.  It should create a 100 satellites in random
positions between leo out to geo each pointing in a random
direction. Call the exclusion function.  For each of these
cases, using plotly, create a plot containing the earth,
the satellites position with a pointing vector pointing away
from it, and vectors to the moon, sun, and earth, together with
an indication if the view was excluded or not.


## [2025-08-15 Fri]

Summary:  I actually did get a nice function to generate evenly
spaced 3d points in, and get it tested.  Working well with github.


Looking at the plan above, I wrote a prompt for gemini to create
the space filling data.
That worked, and I added a function to check it.  There
was a bug in that the radial distribution wasn't applired randomly
in az and el, but gemini found that once I mentioned it.
Checking in with git.

### Prompt for Gemini
I need an algorithm that will create a set of points in 3d space.
Relative to a central point, they should be space logarithmically
spaced in distance from the central point, but equally spaced in
angle in any range of distances. Subject to these constraints the
points should lie between an inner and an outer radius. Find this
algorithm, and if possible give me code to execute it.

take the function we just generated and add a new function that creates
4 plots: first, a 3d plot using plotly that displays the points
(assuming we are in a Jupyter notebook), a plot that histograms the
radii of the points, and plots that display the angular distributions
of the points in terms of latitude and longitude. Display the function
so I can copy it.

## [2025-08-14 Thu]
Ok, lots of today has so far just been figuring out git and github and
emacs and remembering those commands.  I think I just need to download
a nice git single page to put in my desk references.


OK, I'm seeing that I can actually do some editing on this in github 
itslef.  It's OK I guess.  

It's rather interesting to be moving these things around between github
and other locations so quickly, and being able to edit thigns everywhere.

OK, the next action I need to do is to actually get radiometry working,
and stuff like that. 

### Prompt1
Create a function called solarexclusion.
Create an exclusion numpy vector. the same length as the number of
satellites.
Create a function which operates on all the satellites in
the list of satellites in a vectorized manner.
create a vector from the satellite to the sun and the vector
representing the satellite pointing.  If the angle between these
two is less than the solar exclusion angle for the satellite,
place a 1 in the exclusion list, othewise leave it as 0.
Return this vector as well as a vector of the angle from
the function.
\section{Derivation of Search Rate}



Create a test function that prints these two vectors out.





