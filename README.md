This is a project for Lukas Scarfe and Daniel Deiros to generate and investigate Eigenmodes of Turbulence for optical research.

Must have ffmpeg to make videos.


# Things to do

1. ~~Make a .py file for the functions so we can just import.~~ **DONE**
2. ~~Globally adjust output mode phase to be same (relatively) as input mode phase.~~ **DONE**
3. Make oversized (10x) fixed phase screens for different turbulence strengths and resolutions. Save in phase_screens folder or use fixed seed. Suggested Filenames resolutionsize_turbstrength_i.csv, up to you. **Kinda done? Need to think about how to apply this now**
4. Make function to fetch an NxN section of the oversized phasescreens given an x,y coordinate. **n/a**
5. Apply the subsection as the abberation. **not done yet but should be straightforward**


## 2026-01-21

1. ~~PhaseScreenKolmogorov.add_row(inf_ps) ends up just adding 0 phase eventually and the whole phase screen is red. We want to be able to extend them infinitely.~~ **Ok kinda figured stuff out. i was plotting them wrong basically. i think this method is ok but maybe messed up when you add a lot of rows like 10*N. Von Karman seems to be better when adding a lot of rows. Yeah I think we should definitely use Von Karman going forward**
    1. ~~Maybe just go to a big fixed phase screen, don't spend more than 2 hours.~~

2. ~~Save images of a gaussian beam propagated through the channel, final output state while moving the turbulence screens. save the images in /images/subdirectory.~~
    1. Also make videos of different OAM beams, HG modes propagating.
    2. End goal is to to watch a single Eigenmode evolve of the timestep.


## 2026-01-26
1. ~~Create extend_phase_screen function with directions.~~
2. ~~run it with 4 phase screens going in 4 directions, start small for the video because it takes time to make the phase screens.~~
3. ~~Try the videomaking (download ffmpeg) ~~

## 2026-03-04
1. ~~Fix propagate single pixel and ensure that when we extract the output we know which j and i it corresponds to. I believe the outputs are being extracted and appended to FieldOut out of order!~~

## 2026-03-10
1. ~~Revise Fried parameter stuff. Really interesting behaviour when we increase the number of phase screens where the phase screens individually become less strong and then we get eigenmodes for super low turbulence. But the way we have it coded is that when considering the phase screens together they should act as the turbulence with the C_n^2 parameter. idk. then when going from midweak turb to mid turb the phase screens gain phase variance very quickly.~~

## 2026-03-19
1. ~~Looking to implement the sigma^2 parameter (Rytov parameter) from Aaron's paper (in the supplementary material). Still need to revise Fried parameter stuff but it might be fine. Can use their paper too.~~

## 2026-03-23
1. For dataset stuff:
    * ~10 steps with ~4 phase screens (moving in different directions)
    * Can do this with like ~96x96 resolution
    * Things to save:
        * CSV of the complex output mode (i.e. the field. the eigenmode. raw pixel values)
        * Propagate a gaussian through the channel. Save the complex output field of this as well.
        * Desired output of the NN --> a CSV of the output field
        * Possible inputs:
            * First just try to propagate a gaussian. Give it gaussian output.
            * Eigenmode output from a previous time step.\
2. Remember to add total Rytov parameter as well.
3. For Daniel: Set up a py file to save these type of stuff (e.g. Gaussian output, separate the real and imaginary (necessary for NN), take the first first four eigenmodes (i.e. the ones with the highest eigenvalue), pad the filename it with a few zeros to go up to like 10000, save the parameters somewhere (txt file or something)). Take 10 steps.
4. Something to add: block printing stuff between timesteps...