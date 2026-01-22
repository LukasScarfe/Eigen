This is a project for Lukas Scarfe and Daniel Deiros to generate and investigate Eigenmodes of Turbulence for optical research.



# Things to do

1. ~~Make a .py file for the functions so we can just import.~~ **DONE**
2. ~~Globally adjust output mode phase to be same (relatively) as input mode phase.~~ **DONE**
3. Make oversized (10x) fixed phase screens for different turbulence strengths and resolutions. Save in phase_screens folder or use fixed seed. Suggested Filenames resolutionsize_turbstrength_i.csv, up to you. **Kinda done? Need to think about how to apply this now**
4. Make function to fetch an NxN section of the oversized phasescreens given an x,y coordinate. **n/a**
5. Apply the subsection as the abberation. **not done yet but should be straightforward**


## 21-01-26

1. PhaseScreenKolmogorov.add_row(inf_ps) ends up just adding 0 phase eventually and the whole phase screen is red. We want to be able to extend them infinitely.
    1. Maybe just go to a big fixed phase screen, don't spend more than 2 hours.

2. Save images of a gausian beam propagated through the channel, final output state while moving the turbulence screens. save the images in /images/subdirectory.
    1. End goal is to to watch a single Eigenmode evolve of the timestep. 