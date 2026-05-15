# Using YAML file to acquire parameters for data generation

if __name__ == "__main__":
    import yaml
    from functions.colours          import *
    from functions.measurement      import *
    from functions.optical_modes    import *
    from functions.plotting         import *
    from functions.propagation      import *
    from functions.turbulence       import *
    from functions.eigenmodes       import *

    from LightPipes                 import *

    pmap, imap, customColoursBGY, customColoursViridis = colours()

    # Reading parameters from YAML file
    with open('parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    print("Parameters read from 'parameters.yaml':")

    #  Create dataset folder if not made already
    os.makedirs(parameters["dataset_folder_name"], exist_ok=True)

    # Define data generation function
    def data_gen(size, N, lensSize, wavelength, w0, z, num_phase_screens, C2_n, TurbStrength, shift, num_of_fluctuations, start_point,
                phase_screen_seed, Rytov_total, Rytov_part, dataset_collect, plotting, dataset_folder_name):

        # Pre-define variables
        abbs = []
        allEigenBeams = []
        allEigenBeamsPropagated = []

        for i in range(start_point):
            abbs = gen_turb_channel(size, N, wavelength, z, C2_n[TurbStrength], num_phase_screens, phase_screen_seed, shift, abbs)

        for i in progress(range(start_point,start_point+num_of_fluctuations+1,1), desc="Generating data..."):
            # Update user
            # print(f"Working on timestep {i+1} of {start_point+num_of_fluctuations+1}")
            if i == 0: # only on first step
                # Generate turbulent channel
                abbs = gen_turb_channel(size, N, wavelength, z, C2_n[TurbStrength], num_phase_screens, phase_screen_seed, shift, abbs)
            # Extract raw phase screen array
            abbs_screens = extract_screens(abbs)
            # Propagate single pixels through channel
            if __name__ == "__main__":
                FieldsOut, end_fields = parallelpropagatePixels(size, wavelength, N, z, lensSize, abbs_screens)
                # print(f"Successfully processed {len(end_fields)} pixels.")
            # Calculate eigenvalues and eigenvectors
            eigVals, eigVecs, eigMags = eigen_vals_vecs(end_fields)
            # Determine eigenmodes
            eigenBeams, eigenBeamsPropagated = eigenmodes(size, wavelength, N, z, eigVecs, abbs)
            # Append eigenbeams to list # commented out for now
            # allEigenBeams.append(eigenBeams)
            # allEigenBeamsPropagated.append(eigenBeamsPropagated)

            if plotting:
                # plotting eigenbeams before/after propaganda
                plot=plot_beam(eigenBeams[:5]+eigenBeamsPropagated[:5],rows=2,dpi=N*3)
                plt.show();plt.close()

            # Fluctuate turbulent channel
            abbs = gen_turb_channel(size, N, wavelength, z, C2_n[TurbStrength], num_phase_screens, phase_screen_seed, shift, abbs)

            # Propagate Gaussian through the channel
            Gaussian = OAM(Begin(size, wavelength, N), w0, 0)
            Gaussian_prop = propChannel(Gaussian, z, abbs)
            Gaussian_prop_reversed = propChannel(Gaussian,z,abbs[::-1])

            if plotting:
                # Plot before/after propagation of Gaussian for reference
                plot = plot_beam([Gaussian, Gaussian_prop], rows=2)
                plt.show();plt.close()

                # Plot phase screens
                axes=[]
                images=[]
                ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
                fig = plt.figure(1, figsize=(9,3), constrained_layout=True)
                axes=[]
                images=[]
                for j in range(num_phase_screens):
                    ax = fig.add_subplot(1,num_phase_screens,j+1)
                    im = ax.imshow(wrap_to_pi(abbs[j].scrn), cmap=pmap, vmin=-np.pi, vmax=np.pi)
                    axes.append(ax)
                    images.append(im)
                cbar = fig.colorbar(im)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
                plt.show();plt.close()

            if dataset_collect:
                # Define folder name for time step
                dataset_folder_name_t = f"{dataset_folder_name}/t_{i+1:05d}"
                # Create folder if not made already
                os.makedirs(dataset_folder_name_t, exist_ok=True)
                os.makedirs(dataset_folder_name_t, exist_ok=True)

                # Save output Gaussian beam
                np.savetxt(f"{dataset_folder_name_t}/gaussian_prop_forward_real.csv", Gaussian_prop.field.real, delimiter=",")
                np.savetxt(f"{dataset_folder_name_t}/gaussian_prop_forward_imag.csv", Gaussian_prop.field.imag, delimiter=",")

                np.savetxt(f"{dataset_folder_name_t}/gaussian_prop_reversed_real.csv", Gaussian_prop_reversed.field.real, delimiter=",")
                np.savetxt(f"{dataset_folder_name_t}/gaussian_prop_reversed_imag.csv", Gaussian_prop_reversed.field.imag, delimiter=",")

                # Save best four eigenmodes
                for j in range(4):
                    np.savetxt(f"{dataset_folder_name_t}/eigenmode_{j+1:03d}_real.csv", eigenBeams[j].field.real, delimiter=",")
                    np.savetxt(f"{dataset_folder_name_t}/eigenmode_{j+1:03d}_imag.csv", eigenBeams[j].field.imag, delimiter=",")

                # Save parameters into a textfile (only once)
                if i==0: # First time step
                    with open(f"{dataset_folder_name}/parameters.txt", "w") as file:
                        file.write(f"Window size: {size} m\n")
                        file.write(f"Resolution (NxN): {N}\n")
                        file.write(f"Lens radius: {lensSize} m\n")
                        file.write(f"Wavelength: {wavelength} m\n")
                        file.write(f"Beam radius: {w0} m\n")
                        file.write(f"Total propagation distance: {z} m\n")
                        file.write(f"Number of phase screens: {num_phase_screens}\n")
                        file.write(f"C^2_n: {C2_n[TurbStrength]}\n")
                        file.write(f"Pixels shifted per time step: {shift}\n")
                        file.write(f"Phase screen seed: {phase_screen_seed}\n")
                        file.write(f"Total Rytov parameter: {Rytov_total}\n")
                        file.write(f"Partial Rytov parameter: {Rytov_part}\n")
                        file.write(f"Total timesteps: {num_of_fluctuations+1}\n")

            # print top 5 eigenvalue magnitudes
            # print(f"Eigenvalue magnitudes: {eigMags[:5]}")

    # Generate data
    data_gen(**parameters)
