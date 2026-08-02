import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from minimalsimulation import create_empty_simulation
from observatories import add_observatories, propagate_observatories
from celestialbodies import add_celestial_bodies, celestial_update
from detector import makeDetector
from exclusion import exclusion
from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

def run_observatory_animation():
    print("=== Generating Ground Observatory 3D Animation ===")
    
    # 1. Initialize Simulation
    start_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time, delta_time=3600.0) # 1 hour steps
    
    # Observatory at Table Mountain, CA (lat=34.38, lon=-117.68, alt=2286m)
    latitudes = np.array([34.38])
    longitudes = np.array([-117.68])
    altitudes = np.array([2286.0])
    
    add_observatories(sim_data, num_observatories=1, latitudes=latitudes, longitudes=longitudes, altitudes=altitudes)
    add_celestial_bodies(sim_data)
    
    # Configure detector with 10 deg horizon limit (earthEx)
    d = makeDetector(1, band='V', fov=np.radians(15), ifov=np.radians(0.15), aper=1.0,
                     category=['observatories'], asset_index=np.array([0], dtype=int))
    d.earthEx = np.array([np.radians(10.0)])
    sim_data.detector = d
    
    # 2. Define 3 Space Targets in GCRS (ECI)
    # Target 1: In GEO directly above the observatory at start time
    propagate_observatories(sim_data, start_time)
    obs_pos_init = sim_data.observatories.position[0]
    target_geo = (obs_pos_init / np.linalg.norm(obs_pos_init)) * 42164000.0 # GEO orbit radius
    
    # Target 2: Static GCRS target at GEO distance on x-axis (will rise/set)
    target_rising = np.array([42164000.0, 0.0, 0.0])
    
    # Target 3: Behind Earth (relative to initial obs position)
    target_blocked = -target_geo
    
    targets = np.vstack([target_geo, target_rising, target_blocked])
    target_names = ["GEO (Zenith Lock)", "Inertial Star (Rises/Sets)", "Hidden Target (Behind Earth)"]
    
    # Scaling factor for plotting (so the Earth and GEO are in the same frame without being too tiny)
    scale = 1e6 # Plot in thousands of km
    
    # 3. Simulate over 24 hours to generate animation frames
    frames = []
    times = []
    
    for step in range(24):
        current_time = start_time + timedelta(hours=step)
        times.append(current_time.strftime("%H:%M UTC"))
        
        # Propagate observatory and celestial bodies
        celestial_update(sim_data, current_time)
        propagate_observatories(sim_data, current_time)
        
        obs_pos = sim_data.observatories.position[0]
        zenith = obs_pos / np.linalg.norm(obs_pos)
        
        # Determine visibility for each target
        pointing_lines_x = []
        pointing_lines_y = []
        pointing_lines_z = []
        line_colors = []
        
        for idx, targ in enumerate(targets):
            # Point detector at this target
            pointing_vec = targ - obs_pos
            sim_data.detector.pointing[0] = pointing_vec / np.linalg.norm(pointing_vec)
            
            # Check exclusions (below horizon / Sun / Moon)
            is_excluded = exclusion(sim_data, 0)
            color = "red" if is_excluded else "green"
            
            # Add line from observatory to target
            pointing_lines_x.extend([obs_pos[0]/scale, targ[0]/scale, None])
            pointing_lines_y.extend([obs_pos[1]/scale, targ[1]/scale, None])
            pointing_lines_z.extend([obs_pos[2]/scale, targ[2]/scale, None])
            line_colors.append(color)
            
        frames.append({
            'obs_pos': obs_pos / scale,
            'zenith': zenith,
            'lines_x': pointing_lines_x,
            'lines_y': pointing_lines_y,
            'lines_z': pointing_lines_z,
            'colors': line_colors,
            'time': current_time.strftime("%Y-%m-%d %H:%M UTC")
        })

    # 4. Construct Plotly Figure
    fig = go.Figure()
    
    # Draw Earth as a sphere
    r_earth = 6378137.0 / scale
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = r_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = r_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = r_earth * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', showscale=False, opacity=0.3,
        name='Earth'
    ))
    
    # Draw Targets
    fig.add_trace(go.Scatter3d(
        x=targets[:,0]/scale, y=targets[:,1]/scale, z=targets[:,2]/scale,
        mode='markers+text',
        marker=dict(size=8, color='gold', symbol='diamond'),
        text=target_names, textposition="top center",
        name='Targets'
    ))
    
    # Draw initial Observatory
    fig.add_trace(go.Scatter3d(
        x=[frames[0]['obs_pos'][0]], 
        y=[frames[0]['obs_pos'][1]], 
        ysrc=None,
        z=[frames[0]['obs_pos'][2]],
        mode='markers+text',
        marker=dict(size=10, color='darkblue', symbol='circle'),
        text=['Table Mountain Observatory'], textposition="bottom center",
        name='Observatory'
    ))
    
    # Draw initial Line of Sight beams
    for i in range(3):
        lx = frames[0]['lines_x'][i*3:i*3+2]
        ly = frames[0]['lines_y'][i*3:i*3+2]
        lz = frames[0]['lines_z'][i*3:i*3+2]
        color = frames[0]['colors'][i]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode='lines',
            line=dict(color=color, width=3, dash='dash' if color == 'red' else 'solid'),
            name=f'LOS to {target_names[i]}'
        ))
        
    # Set up layout and animation buttons
    fig.update_layout(
        title="24-Hour Ground Observatory Tracking Movie (GCRS Frame)",
        scene=dict(
            xaxis=dict(title='X (1000s km)', range=[-50, 50]),
            yaxis=dict(title='Y (1000s km)', range=[-50, 50]),
            zaxis=dict(title='Z (1000s km)', range=[-50, 50]),
            aspectmode='cube'
        ),
        width=1000, height=800
    )
    
    # Add frames for animation
    plotly_frames = []
    for step_idx, f in enumerate(frames):
        frame_data = [
            # Earth (Surface is trace 0, stays static)
            fig.data[0],
            # Targets (Static positions, trace 1)
            fig.data[1],
            # Observatory moves (trace 2)
            go.Scatter3d(
                x=[f['obs_pos'][0]], y=[f['obs_pos'][1]], z=[f['obs_pos'][2]],
                mode='markers+text',
                marker=dict(size=10, color='darkblue'),
                text=['Table Mountain Observatory'], textposition="bottom center"
            )
        ]
        # Adding individual line traces (traces 3, 4, 5)
        for i in range(3):
            lx = f['lines_x'][i*3:i*3+2]
            ly = f['lines_y'][i*3:i*3+2]
            lz = f['lines_z'][i*3:i*3+2]
            color = f['colors'][i]
            frame_data.append(go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode='lines',
                line=dict(color=color, width=3, dash='dash' if color == 'red' else 'solid')
            ))
            
        plotly_frames.append(go.Frame(
            data=frame_data,
            name=f['time']
        ))
        
    fig.frames = plotly_frames
    
    # Play and Pause buttons
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1, xanchor="right", y=0, yanchor="top"
        )]
    )
    
    # Add a slider to step through hours
    sliders = [dict(
        steps=[dict(
            method="animate",
            args=[[f['time']], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=0))],
            label=f['time']
        ) for f in frames],
        transition=dict(duration=0),
        x=0.1, y=0, currentvalue=dict(font=dict(size=12), prefix="Epoch: ", visible=True, xanchor="right"),
        len=0.9
    )]
    fig.update_layout(sliders=sliders)
    
    # Save the dynamic interactive animation as an HTML file
    fig.write_html("observatory_scan_movie.html")
    print("Saved 3D animation to: observatory_scan_movie.html")
    return fig

if __name__ == '__main__':
    run_observatory_animation()
