import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp2d
from scipy.integrate import cumtrapz
from typing import List
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

    # some default values for the vehicle sizes
    length = 5
    width = 2


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

        if type(time) is float:
            # if one singular timestep is given, change to minimal np array
            time = np.array([0, time])

        #calculate the argument of the trigonometric functions first
        angle = self.yaw + yaw_rate_candidate*time
        speed = self.v + self.a*time
        
        # then integrate the position
        x = cumtrapz(speed*time[1]*np.cos(angle)) + self.x
        y = cumtrapz(speed*time[1]*np.sin(angle)) + self.y

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

    
    def plotoutline(self):
        pass




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
            copy.update_position(float(search_time[1]))
            potential = copy.get_risk_potential(x,y)
            result[i] = potential
        return result

        


class Lane(HazardSource):
        """Hazard source for adapted for road hazard"""
        def __init__(self, x_lane_center, y_lane_center) -> None:

            # set values to recomendation for road type
            self.weight = -7.41e4
            self.variance_x = 2**2#2.40**2
            self.variance_y = 2**2#2.40**2

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
            risk_potential -= self.risk_function(0, Y-points_interp)
                
            return risk_potential
        
        





class PotentialFieldMethod:
    """
    The potential field method is a path finding algorithm for autonomous vehicles<br>
    this is a blueprint, that provides the basic structure of the algorithm based on 
    """
    
    def __init__(self) -> None:

        # state of the ego vehicle
        self.ego = VehicleModel()
        self.ego.set_position(0,0)

        # containers for obstacles and road    
        self._obstacles:List[MovingObstacle] = []
        self._road:List[Lane] = []
        self._target:HazardSource = None

        self.v_target = 50/3.6
        self.a_max = 3

        # parameters that define the driving behaviour
        self.yaw_rate_candidates = np.linspace(-0.5,0.5,35)
        self.search_time = np.linspace(0,1,18)
        self.performance_weights = np.ones(self.search_time.shape)
        self.q = 200 # weight for punishing aggressive steering

        

        # save the results in these variables
        self.performance = 0
        self.ideal_yaw_rate = None
        self.ideal_trajectory = None
        self.ideal_acceleration = None

        # additional calculation values
        self.use_dynamic_model = False
        self.use_longitudal_control = False
        self.dt = 0
        self.p = 0
        self.d = 1
        self.zone_values = np.zeros((2))


    def get_risk_potential(self, X, Y):
        """calculates the total risk as a sum of the risk potential of all given osi objects"""
        result = np.zeros_like(X, dtype=np.float32)

        # now add risk potential for all saved objects
        for ob in self._obstacles:
            result += ob.get_risk_potential(X, Y)

        # and the road itself
        for lane in self._road:
            #calculate risk potential along lane and subtract it from the result
            result -= lane.get_risk_potential(X, Y)

        if self._target is not None:
            result += self._target.get_risk_potential(X, Y)
            
        return result
    

    

    

    def performance_index(self, yaw_rate_candidate):
        """Calculates the performance J for a given yaw rate.\n
        The performance J and the correlating trajectory are returnd as a list [J, trajectory]
        """

        trajectory = self.ego.predict_position(yaw_rate_candidate, self.search_time)

        # original would be np.sum, but np. average should allow for better comparability
        J = np.average((self.get_risk_potential(*trajectory) + self.q*yaw_rate_candidate**2)*self.performance_weights[1:])

        return J
    

    def find_ideal_yawrate(self, vizualize_performance=False):
        """Estimates the ideal yaw rate for the Vehicle to optimize risk, whilst reducing steer input."""
        
        #evaluate performance for all yaw rate candidates
        performance = np.array([(self.performance_index(yaw_rate), yaw_rate) for yaw_rate in self.yaw_rate_candidates])

        if self.use_dynamic_model:
            # calculate the difference in performance for each yaw rate candidate
            dynamic_performance = (performance - self.performance)
            # carry over old value to the next step
            self.performance = performance.copy()

            # sum of new and old performance
            performance[:,0] += self.d*dynamic_performance[:,0]

            # vizualize_performance=True
        
        if vizualize_performance:

            performance_viz = np.array(performance)

            plt.plot(performance_viz[:,1],self.performance[:,0], label='static')
            plt.plot(performance_viz[:,1],dynamic_performance[:,0], label='dynamic')
            plt.plot(performance_viz[:,1],performance_viz[:,0], label='combined')
            plt.xlabel("Yaw Rate Candidate [rad/s]")
            plt.ylabel("Performance Index")
            plt.legend()
            plt.show()

        #find the lowest performance index
        result = sorted(performance, key=lambda a: a[0])[0]
        
        #save all the trajectories for analysis purpose
        self.ideal_trajectory = self.ego.predict_position(result[1],self.search_time)
        self.ideal_yaw_rate = result[1]

        return self.ideal_yaw_rate
    


    
    def append_obstacle(self, obstacle:MovingObstacle):
        if None in [obstacle.x, obstacle.y, obstacle.yaw]:
            print("position of obstacle is not defined")
        self._obstacles.append(obstacle)

    
    def append_lane(self, x_lane_center, y_center):
        lane = Lane(x_lane_center, y_center)
        self._road.append(lane)

    
    

    
    

    
    def update(self, time):
        """update ego position and obstacle positions at once"""

        self.dt = time

        
        self.ego.yaw_rate = self.find_ideal_yawrate()
        if self.use_longitudal_control:
            # update speed
            self.ego.a = self.find_ideal_acceleration(time)
        
        # update all vehicles
        for ob in self._obstacles + [self.ego]:
            ob.update_position(time)



    def front_zone(self):
        X = np.linspace(2, 5, 10) + self.ego.x
        Y = np.zeros_like(X) + self.ego.y

        return self.get_risk_potential(X,Y).mean()


    def center_zone(self):
        X = np.linspace(0,0,1) + self.ego.x
        Y = np.linspace(0,0,1) + self.ego.y
        return self.get_risk_potential(X,Y).mean()


    def rear_zone(self):
        X = np.linspace(-5, -2, 10) + self.ego.x
        Y = np.zeros_like(X) + self.ego.y

        return self.get_risk_potential(X,Y).mean()


    def find_ideal_acceleration(self, dt):
        """evaluate the potential field to find ideal yaw rate"""

        #control loop factors

        #calculate current zone values normalized by center zone
        zone_values = np.array([self.front_zone(), self.rear_zone()]) /self.center_zone()

        #get the difference from previous step
        zone_val_dif = (zone_values - self.zone_values)/dt

        #saved values for next step
        self.zone_values = zone_values

        
        speed_dif = self.v_target - self.ego.v

        a = np.clip(self.p*(zone_values[0]-1) + self.d*zone_val_dif[0] + speed_dif, -self.a_max, self.a_max)

        return a




if __name__ == "__main__":

    # create object as trajectory planner
    pfm = PotentialFieldMethod()

    # add obstacle and set its position
    obstacle = MovingObstacle()
    obstacle.set_position(0,0,v=20,yaw_rate=1)

    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    X = np.linspace(-20, 20, 25)
    Y = np.linspace(-10, 10, 25)

    X,Y = np.meshgrid(X,Y)

    # append obstacle to the pfm scanning
    pfm.append_obstacle(obstacle)

    #create straight lane
    x_lane = np.linspace(-10, 200, 100)
    y_lane = np.zeros(np.shape(x_lane))
    pfm.append_lane(x_lane, y_lane)

    x,y = obstacle.predict_position(obstacle.yaw_rate, pfm.search_time)
  
    z = obstacle.get_predictive_risk_potential(x,y,pfm.search_time)

    obstacle.update_position(pfm.search_time)
    Z = obstacle.get_risk_potential(X,Y)

    print(x.shape)
    print(y.shape)
    print(z.shape)


    kwargs = {
    'cmap': plt.cm.jet, 
    # 'vmin': 0,
    # 'vmax': MovingObstacle.weight*1.2,
    'alpha': 0.3,
}
    ax.plot_surface(X,Y,Z, **kwargs)
    ax.scatter(x,y,z[1:], color='black')
    plt.show()


