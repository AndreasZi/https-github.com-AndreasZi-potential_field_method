import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy


from potential_field_method import PotentialFieldMethod, MovingObstacle, Road



# create pfm object
pfm = PotentialFieldMethod()
pfm.set_position(0,0, v=14) #set ego position (speed is in m/s)

#create lane
lane_x = np.linspace(0, 100, 50)
# lane_y = 0.005*lane_x**2
lane_y = np.zeros_like(lane_x)
pfm.append_lane(lane_x, lane_y)

# lane_x = np.linspace(0, 100, 50)
# # lane_y = 0.005*lane_x**2
# lane_y = np.zeros_like(lane_x) - 3.5
# pfm.append_lane(lane_x, lane_y)

#create obstacle
ob1 = MovingObstacle()
# ob1.set_position(40, 5.0, yaw=np.pi/6)
ob1.set_position(35, -3.5)
pfm.append_obstacle(ob1)

# ob2 = MovingObstacle()
# # ob1.set_position(40, 5.0, yaw=np.pi/6)
# ob2.set_position(35 + 80, -3.5)
# pfm.append_obstacle(ob2)



# simulation time
dt = 0.1
t_max = 10
max_frames = int(t_max/dt)


# setup of car states
pfm.set_position(0,-1.75, 0, 14) #set ego position (speed is in m/s)
ob1.set_position(35, -1.75, 0, v=7)
pfm.d=2


# empty list to keep track of results
x_plot = []
y_plot = []
z_plot = []



"""animation"""
# plot the actual data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# set plot ranges
X = np.linspace(0, 150, 25)
Y = np.linspace(-7, 7, 25) # np.arange(-7, 7, 0.1)
X,Y = np.meshgrid(X,Y)

# get hazard map
Z = pfm.overall_risk_potential(X, Y)

# plot repulsive field as stationary background
kwargs = {
    'cmap': plt.cm.jet, 
    'vmin': 0,
    'vmax': MovingObstacle.weight*1.2,
    'alpha': 0.7,
}

ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$y$ [m]')
ax.set_zlabel(r'$U_{risk}$')

ax.set_title('Potential Field Method Demo')


background = ax.plot_surface(X,Y,Z, **kwargs)
ego_path = ax.plot([], [], [], '--', color = 'black')[0]
ego_path.set_zorder(100)

def update(frame):
    global background, z_plot

    pfm.update(dt)

    # append to plot lists
    x_plot.append(pfm.ego.x)
    y_plot.append(pfm.ego.y)
    z_plot.append(pfm.overall_risk_potential(pfm.ego.x, pfm.ego.y))
    # z_plot = pfm.overall_risk_potential(x_plot, y_plot)
    
    # Update surface

    Z = pfm.overall_risk_potential(X, Y)

    background.remove()
    background = ax.plot_surface(X,Y,Z, **kwargs)

    # Update ego path
    ego_path.set_data(np.array(x_plot), np.array(y_plot))
    ego_path.set_3d_properties(np.array(z_plot))
    
    # manually stop the animation
    if frame >= max_frames-1:
        print('stop animation')
        ani.event_source.stop()

    return background, ego_path

#create animation
print('The animation will take', int(t_max/dt), 'frames')
ani = animation.FuncAnimation(fig=fig, func=update, frames=int(t_max/dt), interval=dt*1000, repeat=True)

plt.show()