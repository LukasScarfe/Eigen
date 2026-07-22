# Using YAML file to acquire parameters for data generation

if __name__ == "__main__":
    import yaml
    import shutil
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

    # Helper: save a complex field as amplitude + phase into one .npz
    def save_ap(path, field):
        np.savez(path, amp=np.abs(field), phase=np.angle(field))

    # Helper: save complex-field difference (now - prev) as amplitude + phase
    def save_diff(path, field_now, field_prev):
        d = field_now - field_prev
        np.savez(path, amp=np.abs(d), phase=np.angle(d))

    # Helper: turn an internal probe key into a nicely typeset label for plots
    def pretty_label(name):
        if name == "gaussian":
            return "Gaussian"
        if name.startswith("LG_"):
            tail = name[3:]
            l = -int(tail[3:]) if tail.startswith("neg") else int(tail)
            return f"LG ℓ={l:+d}"
        if name.startswith("HG_"):
            m, n = name[3:].split("_")
            sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            return f"HG{m.translate(sub)}{n.translate(sub)}"
        return name

    # Define data generation function
    def data_gen(size, N, lensSize, wavelength, w0, z, num_phase_screens, C2_n, TurbStrength, shift, num_of_fluctuations, start_point,
                phase_screen_seed, Rytov_total, Rytov_part, dataset_collect, plotting, dataset_folder_name, animate=True, fps=2):

        # Pre-define variables
        abbs = []
        allEigenBeams = []
        allEigenBeamsPropagated = []
        eigenframes = []            # top-4 eigenbeams per timestep, for the end-of-run animation
        probe_frames_fwd = []       # per-timestep list of forward-propagated probe fields (for animation)
        probe_frames_rev = []       # per-timestep list of reversed-propagated probe fields (for animation)
        prev_fwd = None             # previous timestep's forward-propagated probes (for diffs)
        prev_rev = None             # previous timestep's reversed-propagated probes (for diffs)
        ref_vecs = None             # previous timestep's tracked eigenvectors (for mode tracking)
        n_track = 4                 # number of eigenmodes we follow/save/animate
        overlap_history = []        # per-timestep matched overlaps for each track (for the tracking plot)

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

            # Track modes across timesteps so index j follows the same physical mode.
            # First timestep defines the tracks (keep eigenvalue-magnitude order).
            if ref_vecs is not None:
                perm, overlaps = track_modes(ref_vecs, eigVecs, n_track=n_track)
                # Warn (but proceed) if a track's best overlap is weak -> labeling is unreliable
                for j, ov in enumerate(overlaps):
                    if ov < 0.5:
                        print(f"WARNING: timestep {i+1}, eigenmode track {j+1} low overlap {ov:.2f} (mode identity may have swapped)")
                overlap_history.append(overlaps)
                # Reorder eigenvalues/vectors/magnitudes so column j is the mode matched to track j
                eigVecs = eigVecs[:, perm]
                eigVals = eigVals[perm]
                eigMags = eigMags[perm]
                # Pin the global phase of each tracked mode to its reference, so the SAVED
                # eigenmode fields are temporally coherent (not just the intensity animation).
                for j in range(min(n_track, eigVecs.shape[1])):
                    ph = np.angle(np.vdot(ref_vecs[:, j], eigVecs[:, j]))
                    eigVecs[:, j] = eigVecs[:, j] * np.exp(-1j * ph)
            else:
                # First timestep defines the tracks: perfect self-match by construction
                overlap_history.append(np.ones(min(n_track, eigVecs.shape[1])))

            # Remember this timestep's tracked eigenvectors as the reference for the next step
            ref_vecs = eigVecs[:, :n_track].copy()

            # Determine eigenmodes
            eigenBeams, eigenBeamsPropagated = eigenmodes(size, wavelength, N, z, eigVecs, abbs)
            # Append eigenbeams to list # commented out for now
            # allEigenBeams.append(eigenBeams)
            # allEigenBeamsPropagated.append(eigenBeamsPropagated)

            # Keep the top-4 eigenbeams for the end-of-run animation
            eigenframes.append(eigenBeams[:4])

            if plotting:
                # plotting eigenbeams before/after propaganda
                plot=plot_beam(eigenBeams[:5]+eigenBeamsPropagated[:5],rows=2,dpi=N*3)
                plt.show();plt.close()

            # Fluctuate turbulent channel
            abbs = gen_turb_channel(size, N, wavelength, z, C2_n[TurbStrength], num_phase_screens, phase_screen_seed, shift, abbs)

            # Define probe input fields (name -> input Field).
            # l = 0 is the Gaussian, so it is generated once and not duplicated as LG_0.
            probe_inputs = {"gaussian": OAM(Begin(size, wavelength, N), w0, 0)}
            for l in (-2, -1, 1, 2):
                name = f"LG_{'neg' if l < 0 else ''}{abs(l)}"   # LG_neg2, LG_neg1, LG_1, LG_2
                probe_inputs[name] = OAM(Begin(size, wavelength, N), w0, l)
            for m, n in ((2, 2), (1, 2), (2, 1)):
                probe_inputs[f"HG_{m}_{n}"] = HG(Begin(size, wavelength, N), w0, m, n)

            # Propagate each probe forward and reversed through the (fluctuated) channel
            probes_fwd = {name: propChannel(F, z, abbs)       for name, F in probe_inputs.items()}
            probes_rev = {name: propChannel(F, z, abbs[::-1]) for name, F in probe_inputs.items()}

            # Keep this timestep's probe fields (in a fixed order) for the end-of-run animations
            probe_frames_fwd.append([probes_fwd[name] for name in probe_inputs])
            probe_frames_rev.append([probes_rev[name] for name in probe_inputs])

            if plotting:
                # Plot before/after propagation of each probe for reference
                for name, F in probe_inputs.items():
                    plot = plot_beam([F, probes_fwd[name]], rows=2)
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

                # For the first timestep there is no t-1, so diff against self (yields zeros)
                prev_f = prev_fwd if prev_fwd is not None else probes_fwd
                prev_r = prev_rev if prev_rev is not None else probes_rev

                # Save each probe (amplitude + phase) and its change since the previous timestep
                for name in probe_inputs:
                    save_ap  (f"{dataset_folder_name_t}/{name}_forward.npz",       probes_fwd[name].field)
                    save_ap  (f"{dataset_folder_name_t}/{name}_reversed.npz",      probes_rev[name].field)
                    save_diff(f"{dataset_folder_name_t}/{name}_forward_diff.npz",  probes_fwd[name].field, prev_f[name].field)
                    save_diff(f"{dataset_folder_name_t}/{name}_reversed_diff.npz", probes_rev[name].field, prev_r[name].field)

                # Save best four eigenmodes (amplitude + phase)
                for j in range(4):
                    save_ap(f"{dataset_folder_name_t}/eigenmode_{j+1:03d}.npz", eigenBeams[j].field)

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
                        file.write(f"Probe beams: {', '.join(probe_inputs)}\n")
                        file.write("Output format: .npz per beam, arrays 'amp' (np.abs) and 'phase' (np.angle, radians)\n")
                        file.write("Difference files (*_diff.npz): amp/phase of complex field(t) - field(t-1); first timestep is zeros\n")

            # Remember this timestep's probes so the next one can compute the difference
            prev_fwd, prev_rev = probes_fwd, probes_rev

            # print top 5 eigenvalue magnitudes
            # print(f"Eigenvalue magnitudes: {eigMags[:5]}")

        # Animate the evolution of the top-4 eigenmodes (intensity only) over the run
        if animate and eigenframes:
            frames_dir = f"{dataset_folder_name}/_eigen_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for k, beams in enumerate(progress(eigenframes, desc="Rendering eigenmode frames...")):
                fig = plot_beam(list(beams), rows=1, intensity=True, phase=False, dpi=N*3)
                fig.savefig(f"{frames_dir}/frame_{k:04d}.png")
                plt.close('all')   # plot_beam draws on figure 1; clear it between frames
            create__web_movie(frames_dir, f"{dataset_folder_name}/eigenmodes_top4.webm", fps=fps)
            shutil.rmtree(frames_dir)   # remove the PNG frames once the video is made

        # Animate the probe beams (intensity only): one combined 2x4 grid per timestep,
        # each panel labeled with its mode type. Forward and reversed get their own video.
        if animate and probe_frames_fwd:
            probe_labels = [pretty_label(name) for name in probe_inputs]
            for direction, frames in (("forward", probe_frames_fwd), ("reversed", probe_frames_rev)):
                frames_dir = f"{dataset_folder_name}/_probe_{direction}_frames"
                os.makedirs(frames_dir, exist_ok=True)
                for k, beams in enumerate(progress(frames, desc=f"Rendering probe {direction} frames...")):
                    fig = plot_beam(list(beams), rows=2, intensity=True, phase=False, dpi=N*3, titles=probe_labels)
                    fig.savefig(f"{frames_dir}/frame_{k:04d}.png")
                    plt.close('all')   # plot_beam draws on figure 1; clear it between frames
                create__web_movie(frames_dir, f"{dataset_folder_name}/probes_{direction}.webm", fps=fps)
                shutil.rmtree(frames_dir)   # remove the PNG frames once the video is made

        # Plot each tracked mode's match overlap over the run (diagnostic of tracking quality)
        if len(overlap_history) > 1:
            ov = np.array(overlap_history)          # shape (timesteps, n_track)
            steps = np.arange(1, ov.shape[0] + 1)
            fig = plt.figure(figsize=(8, 4))
            for j in range(ov.shape[1]):
                plt.plot(steps, ov[:, j], marker='.', label=f"track {j+1}")
            plt.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label="warn threshold")
            plt.ylim(0, 1.02)
            plt.xlabel("timestep")
            plt.ylabel("match overlap with previous timestep")
            plt.title("Eigenmode tracking overlap over the run")
            plt.legend(loc="lower left", fontsize=8, ncol=2)
            plt.tight_layout()
            fig.savefig(f"{dataset_folder_name}/mode_tracking_overlap.png", dpi=150)
            plt.close('all')

    # Generate data
    data_gen(**parameters)
