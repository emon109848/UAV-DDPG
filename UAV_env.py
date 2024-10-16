import math
import random

import numpy as np


class UAVEnv(object):
  
    #dimention
    height = 12         # Hight of UAV = 12m
    width = 16          # Width of Road = 4*4 = 16m
    length = 600       # Lenth of road = 200m

    #RSU
    num_rsu=2
    rsu_range = 100     #range = 100m
    c_rsu = 1000        #cpu cycle = 1000 cycle/bit
    f_rsu = 10*10**9    #cpu freq = 10 GHz
    t_power_rsu = 10

    #UAV
    num_uav = 2
    uav_mass = 4        # 4kg
    uav_speed = 8       #8 m/s
    uav_range= 30
    energy_uav1= 10
    energy_uav2= 10
    c_uav = 1000
    f_uav = 3*10**9
    t_power_uav = 5

    #vehicle
    max_vehicle = 20
    current_num_vehicle= 20
    vehicle_speed = 9

    B=20*10**6
    p_noisy_los = 10 ** (-13)
    channel_gain_unit = 5 ** (-13)

    time_slot = 2
        
    action_bound = [-1, 1]  # Corresponding to the tahn activation function
   
    
    action_dim_y = num_uav+num_rsu
    action_dim_x = current_num_vehicle
    
    state_dim_y = current_num_vehicle
    state_dim_x = num_rsu+num_uav*2+2
    # action_dim_x = num_uav+num_rsu
    # action_dim_y = current_num_vehicle
    
    # state_dim_x = current_num_vehicle
    # state_dim_y = num_rsu+num_uav*2+2

    def __init__(self):
        
        # x_axis_vehicle = np.random.uniform(low=0, high=self.length, size=(self.current_num_vehicle,))
        # y_axis_vehicle = np.random.uniform(low=0, high=self.width, size=(self.current_num_vehicle,))

        # # Create a 2D array with x-axis and y-axis as columns
        # self.vehicle = np.column_stack((x_axis_vehicle, y_axis_vehicle))
        
        self.vehicle = np.array([
                                [25.2420427, 3.97587394],
                                [433.656674, 9.58035525],
                                [303.770056, 5.02138818],
                                [127.180495, 12.3802961],
                                [171.555285, 2.2551137],
                                [536.058051, 13.8057135],
                                [592.885149, 3.29387604],
                                [375.297752, 10.2229402],
                                [317.910964, 14.8360648],
                                [303.502465, 5.26682436],
                                [451.541409, 8.95882279],
                                [15.8286242, 0.768819851],
                                [283.88731, 0.339190495],
                                [377.462057, 8.03491364],
                                [444.454289, 13.1174324],
                                [137.227415, 7.4172062],
                                [478.386529, 15.4750168],
                                [339.645843, 6.7043634],
                                [123.128491, 1.89511367],
                                [306.643195, 0.819659546],
                                [418.719999, 8.22698969],
                                [313.692566, 5.705804],
                                [539.353512, 8.07785428],
                                [17.4416592, 0.35790154],
                                [180.955265, 14.4248087],
                                [444.786917, 7.52361768],
                                [490.649706, 0.776612877],
                                [519.885563, 2.18784181],
                                [440.204567, 10.2579489],
                                [267.526775, 6.90093343]
                            ])
        
        # task_list = np.random.randint(2097153, 2621440)
        # self.task = np.column_stack((task_list))
        
        # task_dead = np.random.randint(2000, 10000)
        # self.task_deadline = np.column_stack((task_dead))
        
        
        
        #create a 2d array
        arrays = []
        for i in range(self.current_num_vehicle):
            current_array = np.array([
                self.dis_rsu1(self.vehicle[i]),
                self.dis_rsu2(self.vehicle[i]),
                self.dis_uav1(self.vehicle[i]),
                self.energy_uav1,  
                self.dis_uav2(self.vehicle[i]),
                self.energy_uav2, 
                np.random.randint(2097153, 2621440),
                np.random.randint(2000, 10000)
            ])
            arrays.append(current_array)

        # Stack the 1D arrays vertically to create a 2D array
        self.start_state = np.vstack(arrays)


    def reset(self):
        # self.reset_env()
        self.vehicle = np.array([
                                [25.2420427, 3.97587394],
                                [433.656674, 9.58035525],
                                [303.770056, 5.02138818],
                                [127.180495, 12.3802961],
                                [171.555285, 2.2551137],
                                [536.058051, 13.8057135],
                                [592.885149, 3.29387604],
                                [375.297752, 10.2229402],
                                [317.910964, 14.8360648],
                                [303.502465, 5.26682436],
                                [451.541409, 8.95882279],
                                [15.8286242, 0.768819851],
                                [283.88731, 0.339190495],
                                [377.462057, 8.03491364],
                                [444.454289, 13.1174324],
                                [137.227415, 7.4172062],
                                [478.386529, 15.4750168],
                                [339.645843, 6.7043634],
                                [123.128491, 1.89511367],
                                [306.643195, 0.819659546],
                                [418.719999, 8.22698969],
                                [313.692566, 5.705804],
                                [539.353512, 8.07785428],
                                [17.4416592, 0.35790154],
                                [180.955265, 14.4248087],
                                [444.786917, 7.52361768],
                                [490.649706, 0.776612877],
                                [519.885563, 2.18784181],
                                [440.204567, 10.2579489],
                                [267.526775, 6.90093343]
                            ])
        
        # task_list = np.random.randint(2097153, 2621440, self.current_num_vehicle)
        # self.task = np.column_stack((task_list))
        
        # task_dead = np.random.randint(low=2000, high=10000, size=(self.current_num_vehicle,))
        # self.task_deadline = np.column_stack((task_dead))
        
        self.energy_uav1= 100000
        self.energy_uav2= 100000        
        
        #create a 2d array
        arrays = []
        for i in range(self.current_num_vehicle):
            current_array = np.array([
                self.dis_rsu1(self.vehicle[i]),
                self.dis_rsu2(self.vehicle[i]),
                self.dis_uav1(self.vehicle[i]),
                self.energy_uav1,  # Assuming it's a function call, not a variable
                self.dis_uav2(self.vehicle[i]),
                self.energy_uav2,  # Assuming it's a function call, not a variable
                np.random.randint(20971530, 26214400),
                np.random.randint(500, 1000)
            ])
            arrays.append(current_array)

        # Stack the 1D arrays vertically to create a 2D array
        self.start_state = np.vstack(arrays)
        self.current_state = self.start_state
        

        
        return self.start_state



    def step(self, action):  # 0: 选择服务的ue编号 ; 1: 方向theta; 2: 距离d; 3: offloading ratio
        

        # print(action)
        
        # action = (action == action.max(axis=0, keepdims=True)).astype(int)
        self.action = self.transform_array(action)
        # print(self.action)
        is_terminal = False
        reward = 0

        
        for i in range(self.num_rsu+self.num_uav):
            if i < self.num_rsu:
                for j in range(self.current_num_vehicle):
                    if self.action[i][j]==1:
                        reward_rsu  = self.calculateForRSU(i,j)
                        reward = reward+reward_rsu

                
            else:
                for j in range(self.current_num_vehicle):
                    if self.action[i][j]==1:
                        reward_uav,is_terminal = self.calculateForUAV(i,j)
                        reward = reward+reward_uav
                        if is_terminal :
                            return self.current_state, reward, is_terminal
        self.upadate_Observation()               
        return self.current_state, reward, is_terminal

    def calculateForRSU(self,rsu_index, vehicle_index):
        # print(self.current_state[vehicle_index,rsu_index])
        
        if((self.current_state[vehicle_index,rsu_index]>self.rsu_range) or (self.current_state[vehicle_index][6]==0)):
            return -1
        
        c_gain = self.channel_gain_unit/(self.current_state[vehicle_index,rsu_index])
        t_rate = self.B/np.sum(self.action[rsu_index, :])*np.log2((self.t_power_uav*c_gain / self.p_noisy_los))
        
        #will not calculate total delay at once
        up_delay = self.current_state[vehicle_index][6]/t_rate*1000
        ex_delay = self.current_state[vehicle_index][6]*self.c_uav/self.f_uav*1000
        total_delay = up_delay + ex_delay
        
        
        #full task will not be calculated at once
        
        # self.current_state[vehicle_index][6]=0
        if(total_delay<=self.current_state[vehicle_index][7]):
            deadline=self.current_state[vehicle_index][7]

            self.current_state[vehicle_index][6]=0
            # return (self.current_state[vehicle_index][7]-total_delay)/1000
            return 1
        else: 
            # return -total_delay/1000
            return -1
            
    
    def calculateForUAV(self,uav_index, vehicle_index):
        reward=0

        if(self.current_state[vehicle_index,uav_index]>self.uav_range)or (self.current_state[vehicle_index][6]==0):
            reward=-1
        
        if uav_index==2:
            c_gain = self.channel_gain_unit/(self.current_state[vehicle_index,uav_index])
            t_rate = self.B/np.sum(self.action[uav_index, :])*np.log2(self.t_power_uav*c_gain / self.p_noisy_los)
            up_delay = self.current_state[vehicle_index][6]/t_rate
            ex_delay = self.current_state[vehicle_index][6]*self.c_uav/self.f_uav
            total_delay = up_delay + ex_delay
            
            opareting_energy = total_delay*(self.uav_mass*9.8*self.height+.5*self.uav_mass*self.uav_speed*self.uav_speed)/1000
            self.current_state[vehicle_index][3]=self.current_state[vehicle_index][3]-opareting_energy
            terminal=False
            if self.current_state[vehicle_index][3]<0:
                terminal=True
            
            if(total_delay<=self.current_state[vehicle_index][7]):
                deadline=self.current_state[vehicle_index][7]
                self.current_state[vehicle_index][6]=0
                # reward = (self.current_state[vehicle_index][7]-total_delay)/1000
                reward =1
            else: 
                # reward = -total_delay/1000
                reward = -1
                
        
            

        
        else: 
            c_gain = self.channel_gain_unit/(self.current_state[vehicle_index,4])
            t_rate = self.B/np.sum(self.action[uav_index, :])*np.log2(self.t_power_uav*c_gain / self.p_noisy_los)
            up_delay = self.current_state[vehicle_index][6]/t_rate
            ex_delay = self.current_state[vehicle_index][6]*self.c_uav/self.f_uav
            total_delay = up_delay + ex_delay
            
            opareting_energy = total_delay*(self.uav_mass*9.8*self.height+.5*self.uav_mass*self.uav_speed*self.uav_speed)/1000
            self.current_state[vehicle_index][5]=self.current_state[vehicle_index][5]-opareting_energy
            terminal=False
            if self.current_state[vehicle_index][5]<0:
                terminal=True
            
        
            if(total_delay<=self.current_state[vehicle_index][7]):
                deadline=self.current_state[vehicle_index][7]

                self.current_state[vehicle_index][6]=0
                # reward = (self.current_state[vehicle_index][7]-total_delay)/1000
                reward = 1
            else: 
                # reward = -total_delay/1000
                reward =-1
        
            
        return reward,terminal
        
    

    
    def dis_rsu1(self,vehicle):
        target_point = np.array([200, 0])
        distance = np.linalg.norm(vehicle - target_point)
        return distance
    def dis_rsu2(self,vehicle):
        target_point = np.array([400, 0])
        distance = np.linalg.norm(vehicle - target_point)
        return distance
    def dis_uav1(self,vehicle):
        target_point = np.array([100, 8])
        distance = np.linalg.norm(vehicle - target_point)
        return distance
    def dis_uav2(self,vehicle):
        target_point = np.array([500, 8])
        distance = np.linalg.norm(vehicle - target_point)
        return distance
    
    # def upadate_Observation(self):
    #     self.update_vehicle_position()
    #     #create a 2d array
    #     arrays = []
    #     for i in range(self.current_num_vehicle):
    #         current_array = np.array([
    #             self.dis_rsu1(self.vehicle[i]),
    #             self.dis_rsu2(self.vehicle[i]),
    #             self.dis_uav1(self.vehicle[i]),
    #             self.energy_uav1,  # Assuming it's a function call, not a variable
    #             self.dis_uav2(self.vehicle[i]),
    #             self.energy_uav2,  # Assuming it's a function call, not a variable
    #             np.random.randint(2097153, 2621440),
    #             np.random.randint(2000, 10000)
    #         ])
    #         arrays.append(current_array)

    #     # Stack the 1D arrays vertically to create a 2D array
    #     self.current_state = np.vstack(arrays)
        
    
    def upadate_Observation(self):
        column_index=0
        increase_value = self.vehicle_speed
        array = self.vehicle
        array[:, column_index] += increase_value
        # array[:, column_index] = np.where(array[:, column_index] > 600, 0, array[:, column_index])
        self.vehicle= array
        
        arrays = []
        # for i in range(self.current_num_vehicle):
        #     current_array = np.array([
        #         self.dis_rsu1(self.vehicle[i]),
        #         self.dis_rsu2(self.vehicle[i]),
        #         self.dis_uav1(self.vehicle[i]),
        #         self.energy_uav1,  # Assuming it's a function call, not a variable
        #         self.dis_uav2(self.vehicle[i]),
        #         self.energy_uav2,  # Assuming it's a function call, not a variable
        #         np.random.randint(20971530, 26214400),
        #         np.random.randint(500, 1000)
        #     ])
        #     arrays.append(current_array)

        # # Stack the 1D arrays vertically to create a 2D array
        # self.start_state = np.vstack(arrays)
        # self.current_state = self.start_state
        
        for i in range(self.current_num_vehicle):
        # Check the condition for the 1st column
            if self.vehicle[i, 0] > 600:
                # If the condition is true, replace with 0 and define task size and task delay
                self.vehicle[i, 0] = 0
                task_size = np.random.randint(20971530, 26214400)
                task_delay = np.random.randint(500, 1000)
                energy_uav1=self.current_state[i, 3]
                energy_uav2=self.current_state[i, 5]
            else:
                # If the condition is false, keep the previous value and use the existing task size and task delay
                task_size = self.current_state[i, 6]
                task_delay = self.current_state[i, 7]
                energy_uav1=self.current_state[i, 3]
                energy_uav2=self.current_state[i, 5]
                
            current_array = np.array([
            self.dis_rsu1(self.vehicle[i]),
            self.dis_rsu2(self.vehicle[i]),
            self.dis_uav1(self.vehicle[i]),
            energy_uav1,  # Assuming it's a function call, not a variable
            self.dis_uav2(self.vehicle[i]),
            energy_uav2,  # Assuming it's a function call, not a variable
            task_size,
            task_delay
            ])
            arrays.append(current_array)

        # Stack the 1D arrays vertically to create a 2D array
        self.current_state = np.vstack(arrays)
  
        
    @staticmethod
        
    def transform_array(arr):
        # Replace -1 <= value < 0 with 0, else with 1
        transformed_arr = np.where((arr >= -1) & (arr < 0), 0, 1)

        # Find columns with multiple 1s
        multiple_ones_cols = np.where(transformed_arr.sum(axis=0) > 1)[0]

        # Iterate over columns with multiple 1s and keep the one with the highest original value
        for col in multiple_ones_cols:
            row_indices = np.where(transformed_arr[:, col] == 1)[0]
            max_val_index = np.argmax(arr[row_indices, col])
            transformed_arr[row_indices[row_indices != row_indices[max_val_index]], col] = 0

        return transformed_arr
        

