# Import routines

import numpy as np
import math
import random

from itertools import product

# Defining hyperparameters
m = 5 # number of locations, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
      
        self.action_space = self.init_action_space()      
        self.state_space = self.init_state_space()
        self.state_init = self.state_space[np.random.choice(len(self.state_space))]

        # Start the first round
        self.reset()
        
        
    def init_action_space(self):
        """ The action space A will be: (𝑚 − 1) ∗ 𝑚 + 1 for m locations. 
            The '+ 1' is for action (0,0) 
            Each action will be a tuple of size 2.
            Possible actions would be of the form: (𝑖, 𝑗) where i and j can be any location but i ≠ j
            (0, 0) tuple will represents ’no-ride’ action.
        """
        
        all_action_space = [(i,j) for i in range (1, m + 1) for j in range (1, m + 1) if i != j]
        
        all_action_space.append((0,0))
        
        return all_action_space
        
        
    def init_state_space(self):
        """The state space is defined by the driver’s current location along with 
           the time components (hour of the day and the day of the week).
           𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘 𝑤ℎ𝑒𝑟𝑒 𝑖 = 1 … 𝑚; 𝑗 = 0 …  𝑡 − 1; 𝑘 = 0 …  𝑑 − 1
           The complete state space is a product of number of locations, 
           number of days in a week and number of hours in a day.
           """
        
        number_locations = [i for i in range(1, m + 1)]
        days_in_a_week = [i for i in range(0, d)]
        hours_in_a_day = [i for i in range(0, t)]  
        
        all_state_space = product(number_locations, hours_in_a_day, days_in_a_week)
        
        # convert to a list using iterator returned by 'product'
        return list(all_state_space)
    
    
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """ Convert the state into a vector so that it can be fed to the NN. 
            This method converts a given state into a vector format. 
            The vector is of size m + t + d.
        """
        
        current_location, current_hour, current_day = state
        
        all_locations = np.zeros(m)        
        #Location has been seeded from 1 in init so we do '-1' 
        #to get array index
        all_locations[current_location - 1] = 1
        
        hours_in_a_day = np.zeros(t)
        hours_in_a_day[current_hour] = 1
        
        days_in_a_week = np.zeros(d)
        days_in_a_week[current_day] = 1
        
        state_encod = np.hstack((all_locations, hours_in_a_day, days_in_a_week)).reshape(1, m+t+d)
        
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        
        location = state[0]
        
        # The lambda distribution at each location
        # The number of requests (possible actions) at a state is dependent on the location.
        request_at_location = [2,12,4,7,8]
        
        #location m is between 1..m and hence doing '-1' to get index
        requests = np.random.poisson(request_at_location[location - 1])
       
        # The upper limit on customer requests is 15 
        if requests > 15:
            requests = 15
            
        # select random sample of requests from the total action space
        # we will remove no - ride option
        
        total_action_space = self.action_space[:-1]

        # (0,0) is not considered as customer request. 
        possible_actions_index = random.sample(range(len(total_action_space)), requests) 
        actions = [total_action_space[i] for i in possible_actions_index]
        
        # (0,0) is driver's option to go 'offline'
        actions.append([0,0])
        
        # if no actions selected, set possible action index to 0
        # for catering to driver's (0,0) action
        if not possible_actions_index:
            possible_actions_index = [0]
        else:    
            possible_actions_index.append(len(possible_actions_index))


        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward
        
           The reward function will be (revenue earned from pickup point 𝑝 to drop point 𝑞) - (Cost of
           battery used in moving from pickup point 𝑝 to drop point 𝑞) - (Cost of battery used in moving from
           current point 𝑖 to pick-up point 𝑝).
           
           The cost C and the revenue R are purely functions of time, i.e. for every hour of driving, the cost (of
           battery and other costs) and the revenue (from the customer) is the same - irrespective of the traffic
           conditions, speed of the car etc 
        """
        
        current_location, current_hour, current_day = state
        
        pickup_from, drop_to = action 
        
      
        
        #If driver went offline, reward is -C
        if action == (0,0):
            reward = -C
        else:           
            # Location 'm' is starting from 1 so doing '-1' to prevent out of bounds access for 
            # pickup and drop index in Time_matrix which are zero based
            
            
            #If current_location is same as pick up location, the time taken to reach customer will be zero 
            if current_location == pickup_from:
                time_to_customer = 0
                trip_time = Time_matrix[pickup_from - 1, drop_to - 1, current_hour, current_day]
            else:    
                time_to_customer = Time_matrix[current_location - 1, pickup_from - 1, current_hour, current_day]
                
                #The time at customer place might have caused the current hour and day to change.
                time_at_customer_location = int(current_hour + time_to_customer)
                
                (hour_at_customer_location, day_at_customer_location) = \
                                            self.handle_day_change(time_at_customer_location, current_day)
                           
                trip_time = Time_matrix[pickup_from - 1, drop_to - 1, hour_at_customer_location, day_at_customer_location]
            
            reward = R * trip_time - C * (trip_time - time_to_customer)
            
            
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        current_location, current_hour, current_day = state
        
        # No ride and driver went offline
        # The noride action just moves the time component by 1 hour
        if action == (0,0):
            current_hour = int(current_hour + 1)
            
            # The day possibly could have changed after being offline for 1 hour
            (current_hour, current_day) = self.handle_day_change(current_hour, current_day)
            
            next_state = (current_location, current_hour, current_day)
        else:
            time_to_customer = Time_matrix[current_location - 1, pickup_from - 1, current_hour, current_day]
            
            #The time at customer place might have caused the current hour and day to change.
            pickup_time_at_customer_location = int(current_hour + time_to_customer)
            
            # Factor for any day change after reaching pickup point
            (hour_at_customer_location, day_at_customer_location) = \
                                self.handle_day_change(pickup_time_at_customer_location, current_day)

            trip_time = Time_matrix[pickup_from - 1, drop_to - 1, hour_at_customer_location, day_at_customer_location]
            
            trip_end_time = int(trip_time + hour_at_customer_location)
             
             # Factor for any day change after completing trip    
            (trip_end_hour, trip_end_day) = \
                                self.handle_day_change(trip_end_time, day_at_customer_location)
                
            next_state = (drop_to, trip_end_hour, trip_end_day)
            
        return next_state

    def handle_day_change(self, hour, day):
        hour_of_day = hour % t
                
        day_of_week = (day + hour//t) % d
        
        return hour_of_day, day_of_week


    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    
    if __name__ == 'main':
        c = 
