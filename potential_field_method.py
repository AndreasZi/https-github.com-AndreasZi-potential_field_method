import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from scipy.integrate import cumtrapz
from typing import List, Literal
from copy import deepcopy


# This module is the python implementation of the artificial potential field method for autonomous driving
#
# The basis for this algorithm is the following paper:
# "Intelligent Driving System for Safer Automobiles" by Hideo Inoue, Pongsathorn Raksincharoensak and Shintaro Inoue
#
# the basic initialization of the algorithm usis equations and variables taken from this paper.
# Addiditional Functionality was added as part of my masters thesis.
# In my masters thesis I used this code on a osi based simulation tool, but all references to the ASAM OSI interfaces are excluded here.



class VehicleModel:
    """this is a simplified vehicle dynamics model, to showcase the functionality of the module"""
    x = 0.0
    y = 0.0

    yaw = 0.0
    yaw_rate = 0.0
        
    v = 0.0
    a = 0.0

    # some default values for the vehicle sizes (bmw i530)
    length = 4.91
    width = 1.86


    def set_position(self, x, y, yaw=None, v=None, yaw_rate=None, a=None):
        self.x = x
        self.y = y

        if yaw is not None:
            self.yaw = yaw

        if yaw_rate is not None:
            self.yaw_rate = yaw_rate

        if v is not None:
            self.v = v

        if a is not None:
            self.a = a

    
    def predict_position(self, yaw_rate_candidate, time, set_position = False):
        """predict the points on a trajctory for a fixed yaw rate for a set number of moments in time"""
        if type(time) is not np.ndarray:
            # if one singular timestep is given, change to minimal np array
            time = np.array([0, time], dtype=np.float64)

        #calculate the argument of the trigonometric functions first
        angle = self.yaw + yaw_rate_candidate*time
        speed = self.v + self.a*time
        
        # then integrate the position
        x = cumtrapz(speed*time[1]*np.cos(angle), initial=0) + self.x
        y = cumtrapz(speed*time[1]*np.sin(angle), initial=0) + self.y

        if set_position:
            # update the vehicle with the most recent values
            self.yaw = angle[-1]
            self.v = speed[-1]
            self.x = x[-1]
            self.y = y[-1]
        return np.array([x,y])

    def update_position(self, time):
        """This is a very simplified model to do trajectory modeling"""
        # update position this is done on previous speed
        self.predict_position(self.yaw_rate, time = time, set_position=True)

    def distance_to_car(self, vehicle):
        return np.linalg.norm([self.x - vehicle.x, self.y - vehicle.y])

    def get_radius(self):
        """return the radius of the current trajectory"""
        return self.v*self.yaw_rate
    
    def plotoutline(self, color='black', alpha=0.2, animation=False):
        """add the outline of the vehicle to an axis"""
        x_left = self.x -self.length/2
        y_bottom = self.y -self.width/2
        patch = Rectangle((x_left,y_bottom), self.length, self.width, rotation_point='center')
        patch.set_angle(self.yaw*180/np.pi)
        patch.set_color(color)
        patch.set_alpha(alpha)

        if animation:
            patch.set_animated(True)

        return patch
    
    



class HazardSource:
    """This class is a blueprint for Hazard Sources like cars or other obstacles"""
    # parameters from original paper
    weight = 8.99e4
    variance_x = 27.8**2
    variance_y = 3.05**2

    # center of the hazard
    x = 0
    y = 0
    
    def get_risk_potential(self, X, Y):
        """
        Determine the risk potential at given position.
        X and Y need to be numpy arrays with values of x and y respectively
        """
        # convert to np array
        X = np.array(X)
        Y = np.array(Y)

        
        return self.risk_function(X-self.x, Y-self.y)
    
    def risk_function(self, X, Y):
        return self.weight*np.exp(-(X)**2/(self.variance_x)-(Y)**2/(self.variance_y))

    
    
class MovingObstacle(HazardSource, VehicleModel):
    

    def get_risk_potential(self, X, Y):
        """
        Determine the risk potential at given position.
        X and Y need to be numpy arrays with values of x and y respectively
        """
        # convert to np array
        X = np.array(X)
        Y = np.array(Y)

        # create local coordinates system
        X_local = (X-self.x)*np.cos(self.yaw)+(Y-self.y)*np.sin(self.yaw)
        Y_local = (X-self.x)*np.sin(-self.yaw)+(Y-self.y)*np.cos(self.yaw)
        
        

        result = np.zeros_like(X)

        # define the required variables
        X_obstacle_front = +self.length/2
        X_obstacle_rear = -self.length/2
        
        # seperate the result into 3 cases
        # case 1 is behind the obstacle
        case1 = np.less_equal(X_local, X_obstacle_rear)
        result[case1] = self.risk_function(X_local-X_obstacle_rear, Y_local)[case1]

        # case 2 is along the obstacle, in this section the function acts as a 1dim gaussean function
        case2 = np.logical_and(np.greater(X_local, X_obstacle_rear), np.less(X_local, X_obstacle_front))
        result[case2] = self.risk_function(np.zeros(X.shape), Y_local)[case2]
        
        # same as case 1 but uses the other end of the obstacle as median
        case3 = np.greater_equal(X_local, X_obstacle_front)
        result[case3] = self.risk_function(X_local-X_obstacle_front, Y_local)[case3]

        return result
    

    def get_predictive_risk_potential(self, X, Y, search_time):
        """return future risk potential along route"""

        #create a copy of the current vehicle to not change the original position
        copy = deepcopy(self)
        result = np.zeros_like(search_time)

        for i, (x,y) in enumerate(zip(X,Y)):
            # update position on each step
            potential = copy.get_risk_potential(x,y)
            result[i] = potential
            copy.update_position(search_time[1])
        return result

        


class Lane(HazardSource):
        """Hazard source for adapted for road hazard"""

        # set values to recomendation for road type
        weight = -7.41e4
        variance_x = 2.40**2
        variance_y = 2.40**2
        

        def __init__(self, x_lane_center, y_lane_center, role:Literal["travel", "passing", "oncoming"]) -> None:
            # safe the lane function
            self.role: Literal["travel", "passing", "oncoming"] = role
            if self.role == 'travel':
                self.weight = 1.5*Lane.weight

            # save the lane center
            self.x = np.array(x_lane_center)
            self.y = np.array(y_lane_center)


        def get_risk_potential(self, X, Y):
            """returns the repulsive field for all lanes set with the add_road_section function"""

            # convert to np array
            X = np.array(X, dtype=np.float64)
            Y = np.array(Y, dtype=np.float64)

            # initialize the risk potential at its max value to later subtract lane centers from
            risk_potential = np.zeros_like(X, dtype=np.float64)
            
            # interpolate the lane to fit the shape of X
            points_interp = np.interp(X, self.x, self.y)

            #calculate risk potential along lane and subtract it from the result
            risk_potential += self.risk_function(0, Y-points_interp)
                
            return risk_potential
        
        

class acc_demo:
    def __init__(self, ego_vehicle:VehicleModel) -> None:
        self.ego_vehicle = ego_vehicle

        # targets
        self.lead_vehicle:VehicleModel = None
        self.v_target = 100/3.6

        # acelleration limits
        self.a_max = 4
        self.a_min = - 6

        # speed vs distance threshhold
        self.min_dist = 5
        self.min_time = 0.8 #seconds behind leading car
        self.dist_threshhold = self.min_dist

        # controll values
        self.speed_factor = 1
        self.distance_factor = 1


    def update_acceleration(self):
        # initialize speed difference to target speed

        if self.lead_vehicle is not None and self.lead_vehicle.v < self.v_target:
            # a leading car is present but it is faster than the target speed
            v_dif = self.lead_vehicle.v - self.ego_vehicle.v

            # calculate distance component
            dist = self.ego_vehicle.distance_to_car(self.lead_vehicle)
            if self.min_time*self.ego_vehicle.v < self.min_dist:
                # min distance for control
                self.dist_threshhold = self.min_dist
            else:
                # use min time for control
                self.dist_threshhold = self.min_time*self.ego_vehicle.v

            d_dif = dist - self.dist_threshhold

        else:
            # no leading car is present or it is slower get to target speed
            v_dif = self.v_target - self.ego_vehicle.v

            # influence of distance is zero
            d_dif = 0
            
            
        # control the acceleration based on speed and distance
        a = v_dif*self.speed_factor + d_dif*self.distance_factor

        # clip the result
        self.ego_vehicle.a = np.clip(a, self.a_min, self.a_max)

    



class PotentialFieldMethod:
    """
    The potential field method is a path finding algorithm for autonomous vehicles<br>
    this is a blueprint, that provides the basic structure of the algorithm based on 
    """

    # the potential field algorithm is used lane keeping, and overtake steering
    possible_maneuvers = Literal['follow', 'overtake', 'unimpeded']
    
    def __init__(self) -> None:
        # simulation time values
        self.dt = None

        # current maneuver
        self.maneuver:Literal['follow', 'overtake', 'unimpeded'] = None

        # state of the ego vehicle
        self.ego = VehicleModel()

        # initiate adaptive cruise control
        self.longitudal_control = acc_demo(self.ego)        

        # containers for obstacles and road    
        self.hazard_sources:List[HazardSource] = []
        self.moving_obstacles:List[MovingObstacle] = []

        self.lanes:List[Lane] = []
        self.travel_lane:Lane = None
        self.lane_keeping_mode:Literal['lane', 'lead vehicle'] = 'lane'

        # parameters that define the driving behaviour
        self.yaw_rate_candidates = np.linspace(-0.5,0.5,35)
        self.search_time = np.linspace(0,1,18)
        self.performance_weights = np.ones(self.search_time.shape)
        self.q = 200 # weight for punishing aggressive steering

        # save the results in these variables
        self.performance = 0
        self.ideal_yaw_rate = None
        self.ideal_trajectory = None

        # longitudal control 
        self.use_longitudal_control = False

        # use risk potential correleting to search time
        self.use_predictive_risk = False

        # interpolation of previous get_risk_protential for pd control
        self.use_global_pd = False
        self.d_global = 1

        # difference of performance index for pd control
        self.use_local_pd = False
        self.d_local = 1


    def get_risk_potential(self, X, Y, previous=False, predicive=False):
        """calculates the total risk as a sum of the risk potential of all given osi objects"""
        result = np.zeros_like(X, dtype=np.float32)

        # now add risk potential for all saved objects
        for ob in self.hazard_sources:
            # iterate all obstacles and lanse

            if predicive and hasattr(ob, 'v'):
                # predictive risk potential on moving objects
                result += ob.get_predictive_risk_potential(X, Y, self.search_time)

            elif previous and hasattr(ob, 'v'):
                # do one second ago, so it doesnt have to be divided by dt
                ob_copy = deepcopy(ob)
                ob_copy.update_position(-1)
                result += ob_copy.get_risk_potential(X, Y)

            else:
                # regular mode and lanes
                result += ob.get_risk_potential(X, Y)

        # using difference for pd control
        if self.use_global_pd and not previous:
            previous = self.get_risk_potential(X,Y,previous=True)
            difference = result - previous
            result += difference*self.d_global
            
        return result
    

    

    

    def performance_index(self, yaw_rate_candidate):
        """Calculates the performance J for a given yaw rate.\n
        The performance J and the correlating trajectory are returnd as a list [J, trajectory]
        """

        trajectory = self.ego.predict_position(yaw_rate_candidate, self.search_time)

        # original would be np.sum, but np. average should allow for better comparability
        J = np.average((self.get_risk_potential(*trajectory, predicive=self.use_predictive_risk) + self.q*yaw_rate_candidate**2)*self.performance_weights)

        return J
    

    def find_ideal_yawrate(self):
        """Estimates the ideal yaw rate for the Vehicle to optimize risk, whilst reducing steer input."""
        
        #evaluate performance for all yaw rate candidates
        performance = np.array([(self.performance_index(yaw_rate), yaw_rate) for yaw_rate in self.yaw_rate_candidates])

        if self.use_local_pd:
            # calculate the difference in performance for each yaw rate candidate
            dynamic_performance = (performance - self.performance)
            # carry over old value to the next step
            self.performance = performance.copy()

            # sum of new and old performance
            performance[:,0] += self.d_local*dynamic_performance[:,0]

        #find the lowest performance index
        result = sorted(performance, key=lambda a: a[0])[0]
        
        #save all the trajectories for analysis purpose
        self.ideal_trajectory = self.ego.predict_position(result[1],self.search_time)
        self.ideal_yaw_rate = result[1]

        return self.ideal_yaw_rate
    

    
    def append_obstacle(self, obstacle:MovingObstacle):
        if None in [obstacle.x, obstacle.y, obstacle.yaw]:
            print("position of obstacle is not defined")
        self.moving_obstacles.append(obstacle)

        self.hazard_sources.append(obstacle)

    
    def append_lane(self, lane:Lane, weight_factor=1):
        lane.weight*=weight_factor
        self.lanes.append(lane)
        if lane.role == 'travel':
            self.travel_lane = lane

        self.hazard_sources.append(lane)


    
    def update(self, time):
        """update ego position and obstacle positions at once"""

        self.dt = time
        
        self.ego.yaw_rate = self.find_ideal_yawrate()
        if self.longitudal_control is not None:
            self.longitudal_control.update_acceleration()
        
        # update all vehicles
        for ob in self.moving_obstacles + [self.ego]:
            ob.update_position(time)

        self.check_maneuver()



    def check_maneuver(self):
        
        # sort obstacles by distance in travel direction
        vehicles = sorted(self.moving_obstacles + [self.ego], key= lambda ob: ob.x*np.cos(self.ego.yaw) + ob.y*np.sin(self.ego.yaw))
        # this information will be used to decide on maneuver

        # where is the ego car
        i = vehicles.index(self.ego)

        if self.maneuver == 'unimpeded':
            if i != len(vehicles) - 1:
                # the ego car is not yet the last car
                self.follow(vehicles[i+1])

        # check if overtake is ongoing
        if self.maneuver == 'overtake':
            if i == 0:
                # no car has been overtaken, keep waiting
                pass
            elif i == len(vehicles)-1:
                # ego car has overtaken all cars
                distance_to_following_car = self.ego.distance_to_car(vehicles[i-1])

                if distance_to_following_car > self.longitudal_control.min_dist:
                    # enough distance has been established, end overtake
                    self.maneuver = 'unimpeded'
            else:
                # ego car is neither first nor last car
                distance_to_following_car = self.ego.distance_to_car(vehicles[i-1])
                distance_to_leading_car = self.ego.distance_to_car(vehicles[i+1])

                if distance_to_following_car > self.longitudal_control.min_dist:
                    if distance_to_leading_car > self.longitudal_control.dist_threshhold:
                        # safe distance to both cars is established, end overtake
                        self.follow(vehicles[i+1])




        


    def overtake(self):
        """stop following and initiate overtake"""
        if self.maneuver != 'follow':
            print("the car cant currently overtake, as there is no lead car")
        else:
            print("overtake")

            # change maneuver identifier
            self.maneuver = 'overtake'

            # adjust weight lanes
            self.travel_lane.weight = 1.5*Lane.weight

            # revert weight of lead car
            self.longitudal_control.lead_vehicle.weight = MovingObstacle.weight
            
            # remove lead car from longitudal control
            self.longitudal_control.lead_vehicle = None

        # print([lane.weight for lane in self._road + self.travel_lane])


    def follow(self, vehicle:VehicleModel):
        print("follow")
        # change maneuver identifier
        self.maneuver = 'follow'

        if self.lane_keeping_mode == 'lane':
            # adjust travel lane to be dominant
            self.travel_lane.weight = 2.5*Lane.weight
        else: 
            # adjust weight of lead car to be attracting
            vehicle.weight = - MovingObstacle.weight/4
            
        # add new lead car
        self.longitudal_control.lead_vehicle = vehicle

        # print([lane.weight for lane in self._road + self.travel_lane])

    
    def unimpeded(self):
        print("unimpeded")
        # change maneuver identifier
        self.maneuver = 'unimpeded'

        # adjust travel lane to be dominant
        self.travel_lane.weight = 2.5*Lane.weight





if __name__ == "__main__":

    
    kwargs_imshow = {
        'aspect': 2,
        'origin': 'lower',
        'cmap': plt.cm.jet,
        'alpha': 0.8,
        'vmin': 3*Lane.weight,
        'vmax': MovingObstacle.weight,
    }

    # simulation time
    dt = 0.2 # time resolution
    t_max = 28 # end time
    time = np.arange(0,t_max,dt)

    pfm = PotentialFieldMethod()
    # pfm.use_predictive_risk = True

    # set position of car
    pfm.ego.set_position(0,0, v= 100/3.6, yaw = 0.0)
    
    # road setup with two lanes 7 3.5 meters appart
    lane_x = np.linspace(0, 100, 50)
    lane_y = np.zeros_like(lane_x)
    lane1 = Lane(lane_x, lane_y, 'travel')
    pfm.append_lane(lane1)
        
    #second lane with less weight
    lane2 = Lane(lane_x, lane_y - 3.5, 'passing')
    pfm.append_lane(lane2)

    # traffic participant
    obstacle = MovingObstacle()
    obstacle.set_position(20, 0, v=80/3.6)
    pfm.append_obstacle(obstacle)

    obstacle2 = MovingObstacle()
    obstacle2.set_position(60, 0, v=80/3.6)
    pfm.append_obstacle(obstacle2)


    pfm_adjusted = deepcopy(pfm)
    pfm_adjusted.lane_keeping_mode = 'lead vehicle'

    
    pfm.follow(obstacle)
    pfm_adjusted.follow(pfm_adjusted.moving_obstacles[0])



    """animation"""
    fig, axes = plt.subplots(2, 1)

    # set plot ranges
    X_origin = np.linspace(-20, 120, 40)
    Y_origin = np.linspace(-8, 8, 40)
    x_center = obstacle.x
    extent = [X_origin[0]+x_center, X_origin[-1]+x_center, Y_origin[0], Y_origin[-1]]
    X,Y = np.meshgrid(X_origin + x_center, Y_origin)

    # get hazard map
    Z = [pfm.get_risk_potential(X, Y), pfm_adjusted.get_risk_potential(X, Y)]

    print(Z[1].shape)
    # plot repulsive field as background
    backgrounds  = []
    for i, ax in enumerate(axes):
        backgrounds.append(ax.imshow(Z[i], extent=extent, **kwargs_imshow))

    
    # obstacle outline
    
    models = [[pfm.ego, obstacle, obstacle2], [pfm_adjusted.ego, obstacle, obstacle2]]
    names = ['ego', 'obstacle 1', 'obstacle 2']
    colors = ['black', 'grey', 'blue']

    car_viz  = [[axes[i].add_patch(car.plotoutline()) for car in models[i]] for i, ax in enumerate(axes)]
    # text_params = {'va': 'center', 'family': 'monospace',
    #                 'fontsize': '11'}
    
    # texts = [[ax.text(car.x, car.y,name, color=color, **text_params) for car, name, color in zip(models[ia], names, colors)] for ia, ax in enumerate(axes)]


    def update(frame):
        # prediction step
        yr = pfm.find_ideal_yawrate()
        tr = pfm.ego.predict_position(yr, pfm.search_time)

        # get the background data
        x_center = obstacle.x
        extent = [X_origin[0]+x_center, X_origin[-1]+x_center, Y_origin[0], Y_origin[-1]]

        X,Y = np.meshgrid(X_origin + x_center, Y_origin)
        Z = [pfm.get_risk_potential(X, Y), pfm_adjusted.get_risk_potential(X, Y)]

        # update plots
        for ia, ax in enumerate(axes):
            backgrounds[ia].set_data(Z[ia])
            backgrounds[ia].set_extent(extent)

            # update car rectangles
            for i, model in enumerate(models[ia]):
                car_viz[ia][i].remove()
                car_viz[ia][i] = ax.add_patch(model.plotoutline(color = colors[i], alpha=0.5))
                # texts[ia][i].set_x(model.x + 3.5)
                # texts[ia][i].set_y(model.y)

        # print(frame)
        if frame == 30:
            pfm.overtake()
            pfm_adjusted.overtake()
        if frame == 100:
            pfm.overtake()
            pfm_adjusted.overtake()

        # update simulation
        pfm.update(dt)
        pfm_adjusted.update(dt)


    axes[0].set_title("adjusting lane weight")
    
    axes[1].set_title("adjusting lead car weight")

    for ax in axes:
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")

    #create animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(t_max/dt), interval=dt*1000)

    plt.show()