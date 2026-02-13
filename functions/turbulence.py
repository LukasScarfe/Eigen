# Functions for generating a turbulent channel

# Imports
import numpy as np
from aotools.turbulence.infinitephasescreen import PhaseScreenVonKarman

def gen_turb_channel(size, N, wavelength, z, C2_n, num_phase_screens, phase_screen_seed, shift, abbs=[]):
    # First find r0
    r0 = fried(wavelength, C2_n, z, num_phase_screens)

    # Define outer scale
    L0 = N/2 * size / N * 50 # outer scale

    if abbs==[]:
        # Create initial phase screens if phase screens undefined
        abbs=[PhaseScreenVonKarman(nx_size=N, pixel_scale=size/N, r0=r0, L0=L0,random_seed=phase_screen_seed+i) for i in range(num_phase_screens)]
    else:
        # Extend phase screens if phase screens already defined
        for i,screen in enumerate(abbs):
            extend_phase_screen(screen,direction=i,num_steps=shift)

    return abbs

def fried(wavelength, C2_n, z, num_phase_screens):
    return pow(0.423*pow(2*np.pi/wavelength,2)*3*C2_n*z/num_phase_screens,-3/5)

def extend_phase_screen(screen, direction="down", num_steps=1):

    def add_row_down(screen, num_steps):
        for _ in range(num_steps):
            PhaseScreenVonKarman.add_row(screen)
        return None

    def add_row_left(screen, num_steps=1):
        screen.scrn[:] = np.rot90(screen.scrn[:], k=1)
        for _ in range(num_steps):
            PhaseScreenVonKarman.add_row(screen)
        screen.scrn[:] = np.rot90(screen.scrn[:], k=-1)
        return None
    
    def add_row_up(screen, num_steps=1):
        screen.scrn[:] = np.flipud(screen.scrn) # Flip matrix vertically
        for _ in range(num_steps):
            PhaseScreenVonKarman.add_row(screen) # Add rows
        screen.scrn[:] = np.flipud(screen.scrn) # Flip matrix vertically again
        return None
    
    def add_row_right(screen, num_steps=1):
        screen.scrn[:] = np.rot90(screen.scrn[:], k=-1)
        for _ in range(num_steps):
            PhaseScreenVonKarman.add_row(screen)
        screen.scrn[:] = np.rot90(screen.scrn[:], k=1)
        return None

    if direction == "down" or direction%4 == 0:
        add_row_down(screen, num_steps)
        return None
    
    if direction == "left" or direction%4 == 1:
        add_row_left(screen, num_steps)
        return None
    
    if direction == "up" or direction%4 == 2:
        add_row_up(screen, num_steps)
        return None
    
    if direction == "right" or direction%4 == 3:
        add_row_right(screen, num_steps)
        return None