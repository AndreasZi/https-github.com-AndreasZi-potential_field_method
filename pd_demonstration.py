import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy


from potential_field_method import PotentialFieldMethod, HazardSource, Road


# simulation time
dt = 0.1
t_max = 7
max_frames = int(t_max/dt)

# create pfm object
pfm = PotentialFieldMethod()

#create lane
lane_x = np.linspace(0, 300, 50)
# lane_y = 0.005*lane_x**2
lane_y = np.zeros_like(lane_x)
pfm.append_lane(lane_x, lane_y)

#create obstacle
ob1 = HazardSource()
pfm.append_obstacle(ob1)


# setup of car states
pfm.set_position(0,0, 0, 14) #set ego position (speed is in m/s)
ob1.set_position(100, -1.75, 0, v=-14)

# create second model using new calculation
pfm_mod = copy.deepcopy(pfm)
pfm_mod.use_dynamic_model = True
pfm_mod.d=20

# empty list to keep track of results
x_plot = []
y_plot = []
z_plot = []

x_plot_mod = []
y_plot_mod = []
z_plot_mod = []



"""animation"""
fig = plt.figure(figsize=(10, 6))

# setting the size and arrangement
gs = fig.add_gridspec(1, 2, width_ratios=[1,2])

# adding subplots
ax = fig.add_subplot(gs[0, 1],projection='3d')
ax2 = fig.add_subplot(gs[0, 0])

# set plot ranges
X = np.linspace(0, 100, 25)
Y = np.linspace(-4, 4, 25) # np.arange(-7, 7, 0.1)
X,Y = np.meshgrid(X,Y)

# get hazard map
Z = pfm.overall_risk_potential(X, Y)

# create setup for surface plot
kwargs = {
    'cmap': plt.cm.jet, 
    'vmin': 0,
    'vmax': HazardSource.weight*1.2,
    'alpha': 0.7,
}
background = ax.plot_surface(X,Y,Z, **kwargs)

# unmodified vehicle path
ego_path = ax.plot([], [], [], '--', color = 'black', label="static")[0]
ego_path.set_zorder(100)

# modified vehicle path
ego_path_mod = ax.plot([], [], [], '-', color = 'red', label="dynamic")[0]
ego_path_mod.set_zorder(50)

# secondary graph
ax2.bar(['original','modified'], [0,0])

# labels for the main graph
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$y$ [m]')
ax.set_zlabel(r'$U_{risk}$')
ax.set_zticks([0])
ax.legend()
ax.set_title('Comparison p- vs pd Model')

# labels for the secondary graph
ax2.set_ylabel(r'$U_{risk}$')
ax2.set_title('average risk potential at cars position')

def update(frame):
    global background, z_plot

    # update potential field
    pfm.update(dt)
    pfm_mod.update(dt)

    # collect data for unmodified
    x_plot.append(pfm.x)
    y_plot.append(pfm.y)
    z_plot.append(pfm.overall_risk_potential(pfm.x, pfm.y))

    # collect data for modified
    x_plot_mod.append(pfm_mod.x)
    y_plot_mod.append(pfm_mod.y)
    z_plot_mod.append(pfm.overall_risk_potential(pfm_mod.x, pfm_mod.y))
    
    # Update surface
    Z = pfm.overall_risk_potential(X, Y)
    background.remove()
    background = ax.plot_surface(X,Y,Z, **kwargs)

    # Update ego path
    ego_path.set_data(np.array(x_plot), np.array(y_plot))
    ego_path.set_3d_properties(np.array(z_plot))

    # Update modified ego path
    ego_path_mod.set_data(np.array(x_plot_mod), np.array(y_plot_mod))
    ego_path_mod.set_3d_properties(np.array(z_plot_mod))
    
    # update bar plot
    ax2.clear()
    ax2.bar(['p','pd'],[np.sum(z_plot),np.sum(z_plot_mod)])

    # manually stop the animation
    if frame >= max_frames-1:
        print('stop animation')
        ani.event_source.stop()


#create animation
print('The animation will take', int(t_max/dt), 'frames')
ani = animation.FuncAnimation(fig=fig, func=update, frames=int(t_max/dt), interval=dt*1000, repeat=True)

plt.show()