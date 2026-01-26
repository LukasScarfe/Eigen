# Functions

# Imports
import numpy as np
from LightPipes import *
from LightPipes import Field
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import colorcet as cc
import pylab as pl
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm as progress

## COLOURS
from colours import colours
pmap, imap, _, _ = colours()

# Defines mode indices even about 0 for d dimensions
def ell(d):
    if d%2 == 0:
        l=np.linspace(-np.floor(d/2),np.floor(d/2),d+1,dtype=int)
        l=l[l != 0]
    else:
        l=np.linspace(-np.floor(d/2),np.floor(d/2),d,dtype=int)
    return l

#Phases for arbitrary MUBs in prime dimensions
def MUBphases(d,MUB):
    phi,p=[],[]
    for state in range(d):
        for j in range(d):
            p.append(((2*np.pi/d)*(MUB*(j**2)+state*j)))
        phi.append(p)
        p=[]
    return phi

#Mixing large amount of beams in a loop
def MixManyBeams(beams):
    if len(beams) > 1:
        beams[1]=BeamMix(beams[0],beams[1])
        beams.pop(0)
        MixManyBeams(beams)
    return beams[0]

#OAM beams 
def OAM(F, w0, state: int=0 ,phase=0,amp: float=1.0):
    F=GaussLaguerre(F, w0, p=0, l=state, A=amp*(1/w0)*np.sqrt(2/(np.pi*(math.factorial(abs(state))))), ecs=0)
    F=SubPhase(F,Phase(F)+phase)
    return F

#Arbitrary ANG mode in dimension d
def ANG(F,w0,d,state,norm=0):
    intensityNorm=[1,1,187.3568368182197,302.7415771427128,281.4774347094998,342.95046333063704,1,330.2491174614896,276.25225696328897,1,1,1,1,1,1,1,1,1,1,1,1,1]
    Q,p=[],[]
    l=ell(d)
    
    for i in range(d):
        p.append((2*np.pi/d)*i*state)

    for i in range(d):
        Q.append(OAM(F,w0,l[i],p[i]))

    F=MixManyBeams(Q)

    if norm==1:
        return [F,intensityNorm[d]]
    else:
        return F
    
#Arbitrary MUB in d dimensions {2,3,4,5,7,8}
def ArbMUB(F,w0,d,MUB,state,norm=0):
    intensityNorm=[1,1,187.3568368182197,302.7415771427128,281.4774347094998,342.95046333063704,1,330.2491174614896,276.25225696328897,1,1,1,1,1,1,1,1,1,1,1,1,1]
    Q,p=[],[]
    l=ell(d)
    if d==2:
        p=[[[0,0],[0,np.pi]],[[0,np.pi/2],[0,3*np.pi/2]]][MUB][state]
    elif d==4:
        p=[[[0,0,0,0],[0,np.pi,np.pi,0],[0,0,np.pi,np.pi],[0,np.pi,0,np.pi]],[[0,np.pi/2,np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,0],[0,3*np.pi/2,np.pi/2,0]],[[0,0,3*np.pi/2,np.pi/2],[0,np.pi,np.pi/2,np.pi/2],[0,0,np.pi/2,3*np.pi/2],[0,np.pi,3*np.pi/2,3*np.pi/2]],[[0,3*np.pi/2,0,np.pi/2],[0,np.pi/2,np.pi,np.pi/2],[0,np.pi/2,0,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2]]][MUB][state]
    elif d==8:
        p=[[[0,0,0,0,0,0,0,0],[0,np.pi/2,np.pi/2,np.pi,np.pi/2,np.pi,np.pi,3*np.pi/2],[0,np.pi/2,0,np.pi/2,0,3*np.pi/2,np.pi,np.pi/2],[0,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2,0,np.pi],[0,0,np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,np.pi/2],[0,np.pi/2,0,3*np.pi/2,np.pi/2,0,np.pi/2,np.pi],[0,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2,np.pi],[0,0,0,np.pi,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,0,np.pi,0,np.pi,0,np.pi],[0,3*np.pi/2,np.pi/2,0,np.pi/2,0,np.pi,np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,0,np.pi/2,np.pi,3*np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2,0,0],[0,np.pi,np.pi/2,np.pi/2,0,0,np.pi/2,3*np.pi/2],[0,3*np.pi/2,0,np.pi/2,np.pi/2,np.pi,np.pi/2,0],[0,3*np.pi/2,np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2,0],[0,np.pi,0,0,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2]],[[0,0,np.pi,np.pi,0,0,np.pi,np.pi],[0,np.pi/2,3*np.pi/2,0,np.pi/2,np.pi,0,np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,0,3*np.pi/2,0,3*np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2,np.pi,0],[0,0,3*np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,3*np.pi/2],[0,np.pi/2,np.pi,np.pi/2,np.pi/2,0,3*np.pi/2,0],[0,np.pi/2,3*np.pi/2,np.pi,0,np.pi/2,np.pi/2,0],[0,0,np.pi,0,np.pi/2,np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,np.pi,0,0,np.pi,np.pi,0],[0,3*np.pi/2,3*np.pi/2,np.pi,np.pi/2,0,0,3*np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,0,np.pi/2,0,np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2,np.pi,np.pi],[0,np.pi,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,np.pi/2,np.pi,3*np.pi/2,np.pi],[0,3*np.pi/2,3*np.pi/2,0,0,3*np.pi/2,np.pi/2,np.pi],[0,np.pi,np.pi,np.pi,np.pi/2,3*np.pi/2,np.pi/2,np.pi/2]],[[0,0,0,0,np.pi,np.pi,np.pi,np.pi],[0,np.pi/2,np.pi/2,np.pi,3*np.pi/2,0,0,np.pi/2],[0,np.pi/2,0,np.pi/2,np.pi,np.pi/2,0,3*np.pi/2],[0,0,np.pi/2,np.pi/2,3*np.pi/2,np.pi/2,np.pi,0],[0,0,np.pi/2,3*np.pi/2,np.pi,0,3*np.pi/2,3*np.pi/2],[0,np.pi/2,0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,0],[0,np.pi/2,np.pi/2,0,np.pi,3*np.pi/2,np.pi/2,0],[0,0,0,np.pi,3*np.pi/2,3*np.pi/2,np.pi/2,3*np.pi/2]],[[0,np.pi,0,np.pi,np.pi,0,np.pi,0],[0,3*np.pi/2,np.pi/2,0,3*np.pi/2,np.pi,0,3*np.pi/2],[0,3*np.pi/2,0,3*np.pi/2,np.pi,3*np.pi/2,0,np.pi/2],[0,np.pi,np.pi/2,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi,np.pi],[0,np.pi,np.pi/2,np.pi/2,np.pi,np.pi,3*np.pi/2,np.pi/2],[0,3*np.pi/2,0,np.pi/2,3*np.pi/2,0,3*np.pi/2,np.pi],[0,3*np.pi/2,np.pi/2,np.pi,np.pi,np.pi/2,np.pi/2,np.pi],[0,np.pi,0,0,3*np.pi/2,np.pi/2,np.pi/2,np.pi/2]],[[0,0,np.pi,np.pi,np.pi,np.pi,0,0],[0,np.pi/2,3*np.pi/2,0,3*np.pi/2,0,np.pi,3*np.pi/2],[0,np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi/2,np.pi,np.pi/2],[0,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2,0,np.pi],[0,0,3*np.pi/2,np.pi/2,np.pi,0,np.pi/2,np.pi/2],[0,np.pi/2,np.pi,np.pi/2,3*np.pi/2,np.pi,np.pi/2,np.pi],[0,np.pi/2,3*np.pi/2,np.pi,np.pi,3*np.pi/2,3*np.pi/2,np.pi],[0,0,np.pi,0,3*np.pi/2,3*np.pi/2,3*np.pi/2,np.pi/2]],[[0,np.pi,np.pi,0,np.pi,0,0,np.pi],[0,3*np.pi/2,3*np.pi/2,np.pi,3*np.pi/2,np.pi,np.pi,np.pi/2],[0,3*np.pi/2,np.pi,np.pi/2,np.pi,3*np.pi/2,np.pi,3*np.pi/2],[0,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2,0,0],[0,np.pi,3*np.pi/2,3*np.pi/2,np.pi,np.pi,np.pi/2,3*np.pi/2],[0,3*np.pi/2,np.pi,3*np.pi/2,3*np.pi/2,0,np.pi/2,0],[0,3*np.pi/2,3*np.pi/2,0,np.pi,np.pi/2,3*np.pi/2,0],[0,np.pi,np.pi,np.pi,3*np.pi/2,np.pi/2,3*np.pi/2,3*np.pi/2]]][state][MUB]
    else:
        for j in range(d):
            p.append(((2*np.pi/d)*(MUB*(j**2)+state*j)))
            
    for i in range(d):
        Q.append(OAM(F,w0,l[i],p[i]))
    F=MixManyBeams(Q)

    if norm==1:
        return [F,intensityNorm[d]]
    else:
        return F

#Arbitrary FQubit mode in dimension d
def Fqubit(F,w0,d=3,j=0,k=1,m=0):
    l=ell(d)
    p=(2*np.pi/d)*m
    
    Q=BeamMix(OAM(F,w0,l[j],0),OAM(F,w0,l[k],p))
    return Q

# Keeping phase values between -pi and pi
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 *np.pi) - np.pi

### GENERATING BEAM PLOTS

#Show a plot of a single beam, or an array of beams with phase and intensity
def plotBeam(Fs: list[Field],rows: int=1,aperature: float=0,intensity: bool=True,phase: bool=True,dpi: int=300) -> plt.Figure:
    
    """
    Show a plot of a single beam, or an array of beams with phase and intensity. This plot can be saved as an image. using the filename argument.

    Args:
        Fs: Field from LightPipes or list of Fields.
        rows: Number of rows in the plot grid.
        aperature: Size of aperature to draw around each beam (in metres).
        intensity: Whether to show intensity plot.
        phase: Whether to show phase plot.
        dpi: Dots per inch for the output image.
    """

    if not hasattr(Fs, "__len__"):
        Fs=[Fs]

    totalModes = len(Fs)
    if rows>=totalModes:
        columns=1
        rows=totalModes
    else:
        columns = totalModes//rows + (1 if totalModes%rows else 0)
    Position = range(1,totalModes + 1)

    fig_width = columns
    fig_height = rows

    fig = plt.figure(1,figsize=(fig_width, fig_height),dpi=dpi)

    pixels,size=Fs[0].N,Fs[0].siz

    for index,F in enumerate(Fs):
        I=1-Intensity(1,F)
        Phi=np.mod(Phase(F),2*np.pi)

        ax = fig.add_subplot(rows,columns,Position[index])
        ax.set_facecolor('black')
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.margins(0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        ax.imshow(Phi,cmap=pmap,vmin=0,vmax=2*np.pi,interpolation='None') if phase==True else None
        ax.imshow(I,cmap=imap if phase==True else plt.colormaps['gray_r'] ,vmin=np.min(I),vmax=np.max(I),interpolation='None') if intensity ==True else None

        if aperature:
            centre=(pixels/2-0.5,pixels/2-0.5) if pixels%2 == 1 else (pixels/2,pixels/2)
            circle = plt.Circle(centre,aperature*pixels/size, color='w', fill=False,linewidth=0.5)
            ax.add_patch(circle)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    return fig

# Missing comment?
def plotBeamTransparent(Fs, rows=1, intensity=True, phase=True, dpi=300):
    Fs = Fs if isinstance(Fs, list) else [Fs]
    num_beams = len(Fs)
    cols = (num_beams + rows - 1) // rows
    
    # Use the grid size of the field (e.g., 512)
    N = Fs[0].N 
    
    # Calculate figure size in inches to hit exact pixel targets
    fig_w = (cols * N) / dpi
    fig_h = (rows * N) / dpi
    
    # Create figure with NO frame or padding
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='none')
    
    for i in range(num_beams):
        # Calculate position for each subplot manually to ensure 0 padding
        # [left, bottom, width, height] in normalized coordinates (0 to 1)
        col_idx = i % cols
        row_idx = rows - 1 - (i // cols) # Start from top
        
        ax_pos = [col_idx/cols, row_idx/rows, 1/cols, 1/rows]
        ax = fig.add_axes(ax_pos)
        ax.set_axis_off()
        
        F = Fs[i]
        I = Intensity(F)
        P = np.mod(Phase(F), 2 * np.pi)
        
        
        norm_I = (I - I.min()) / (I.max() - I.min()) if I.max() > I.min() else I
        rgba = pmap(P / (2 * np.pi))
        rgba[..., 3] = norm_I if intensity else 1.0
        
        if not phase: 
            rgba[..., :3] = 0.5
            
        ax.imshow(rgba, interpolation='nearest', aspect='auto')

    return fig

### Overlap integrals and crosstalk

#These are custom functions to calculate the overlap integral between two modes

# This is the overlap integral to check the fidelity between two modes
def overlapInt(F,G):
    F,G=Normal(F),Normal(G)
    Ffield,Gfield=np.conjugate(F.field),G.field
    fieldArr=np.multiply(Ffield,Gfield)
    summed=abs(np.sum(fieldArr))**2
    return summed

#Normalizes the overlap integrals for a tomographic measurement in dimension d. 
def normTomography(ints, d):
    return np.concatenate([
        chunk / chunk.sum() if chunk.sum() != 0 else chunk
        for chunk in np.split(ints, range(d, len(ints), d))
    ])

#Crosstalk of two vectors
def crosstalkVecs(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(abs(np.dot(np.conjugate(F),G))**2)
        C.append(c/sum(c))
    return C

#Calculate full crosstalk of two beam lists
def crosstalk(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(overlapInt(F,G))
        C.append(c/sum(c))
    return C

def tomography(Fs,Gs):
    c,C=[],[]
    for F in Fs:
        c=[]
        for G in Gs:
            c.append(overlapInt(F,G))
        C.append(normTomography(c,math.isqrt(len(Fs))))
    return C

def plotCrosstalk(cross):
    fig = plt.figure(1)
    plt.axis('off')
    plt.imshow(cross, interpolation='none', cmap='viridis', vmin=0,vmax=1)
    return fig

def beamsError(Fs,Gs):
    array=crosstalk(Fs,Gs)
    error=1-np.mean([diag[i] for i,diag in enumerate(array)])
    return error

### Propagation

#Propagates one beam through the channel defined by the distance and the list of abberations that is input (abbs)
#If there are abberations, distance is the distance per abberation
def propChannel(F,distance,abbs=1,mode=0):
    if not hasattr(abbs, "__len__"):
        for i in range(abbs):
            if mode==1:
                F=Forvard(F,distance/abbs)
            elif mode == 2:
                F=Forward(F,distance/abbs,F.siz,F.N)
            else:
                F=Fresnel(F,distance/abbs)

    else:
        F=Fresnel(F,distance/(len(abbs)+1))
        for screen in abbs:
            if not isinstance(screen, np.ndarray):
                screen_real = screen.scrn.copy() # needed for infinite phase screens
            else: 
                screen_real = screen
            F=SubPhase(F,Phase(F)+screen_real)
            if mode==1:
                F=Forvard(F,distance/(len(abbs)+1))
            elif mode == 2:
                F=Forward(F,distance/(len(abbs)+1),F.size,F.N)
            else:
                F=Fresnel(F,distance/(len(abbs)+1))
    return F

def propChannelSteps(F,distance,abbs=1):
    Fs=[]
    if not hasattr(abbs, "__len__"):
        for i in abbs:
            F=Forvard(F,int(distance/abbs))
            Fs.append(F)
            F=Forvard(F,distance)
    else:
        for screen in abbs:
            F=SubPhase(F,Phase(F)+screen)
            F=Forvard(F,int(distance/len(abbs)))
            Fs.append(F)
    return ([F,Fs])

### Parellelization of propagation of all pixels

def propagateSinglePixel(FieldIn: Field,j: int, i:int, N:int, z:float, lensSize:float, abbs) -> tuple:
    """
    Performs the calculation for a single (j, i) pair.

    Args:
        FieldIn: Initial field.
        j (int): Outer loop index.
        i (int): Inner loop index.
        N (int): Grid size.
        z (float): Propagation distance.
        abbs: List of numpy ndarray representing phase screens.

    Returns:
        tuple: (FieldOut, endField_data) for the specific (j, i)
    """

    print('Worker started')    
    
    # 1. Create the intensity matrix
    intens = np.zeros((N, N))
    intens[j][i] = 1

    fieldAtSender=CircAperture(SubPhase(SubIntensity(FieldIn, intens),np.zeros((N,N))),lensSize)

    
    #FieldOut = propChannel(SubIntensity(FieldIn, intens), z, abbs)
    #FieldOut = propChannel(CircAperture(SubIntensity(FieldIn, intens),lensSize), z, abbs)
    FieldOut = CircAperture(propChannel(fieldAtSender, z, abbs),lensSize)

    # 4. Extract data
    endField_data = np.array(FieldOut.field).reshape(N**2)

    return FieldOut,endField_data

def parallelpropagatePixels(FieldIn, N, z,lensSize, abbs):
        
    # Initialize the results lists
    FieldsOut = []
    endFields_data = []
    
    max_workers = os.cpu_count()-2 or 4
    print(f"Using {max_workers} threads to propagate beams simulataneously...")

    tasks = [(j, i) for j in range(N) for i in range(N)]
    ...
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            # Pass the raw data/args, NOT the complex FieldIn object
            executor.submit(propagateSinglePixel, FieldIn, j, i, N, z,lensSize, abbs): (j, i) 
            for j, i in tasks
        }
        
        # Iterate over the results as they complete, showing progress
        results_iterator = progress(
            as_completed(future_to_task), 
            total=len(tasks), 
            desc="Propagating invidual Pixels"
        )
        
        for future in results_iterator:
            try:
                # Get the result from the completed task
                FieldOut,endField_data = future.result()
                
                # Append the results to the main lists
                FieldsOut.append(FieldOut)
                endFields_data.append(endField_data)
                
            except Exception as exc:
                # Print any exceptions that occurred inside the worker function
                j, i = future_to_task[future]
                print(f'\n(j={j}, i={i}) generated an exception: {exc}')
                
    return FieldsOut,np.array(endFields_data)

