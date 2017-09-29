#!/usr/bin/env python

# Copyright (c) 2017 Jonathan Terry, BYU Robotics and Dynamics Lab.
# All rights reserved.

# Redistribution and use in source and binary forms are permitted
# provided that the above copyright notice and this paragraph are
# duplicated in all such forms and that any documentation,
# advertising materials, and other materials related to such
# distribution and use acknowledge that the software was developed
# by the BYU Robotics and Dynamics Lab. The name of the
# BYU Robotics and Dynamics Lab may not be used to endorse or promote products derived
# from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.


import sys
import os
import rospy
import numpy as np
import math
from scipy import linalg

import mpc_solver
import time
from math import pi
import copy
import scipy.io as spicy
from scipy.linalg import expm
import scipy.signal as spicycigs
import optparse

from low_level_control.msg import sensors as sensor
from low_level_control.msg import data_msg as data
from low_level_control.msg import joint_angles_msg_array as j_a_msg
from louie_control.msg import mpc_msg
from louie_control.msg import param_msg
from low_level_control.msg import joint_angles_msg

from kl_dyn_params import params as dyn_params
from kl_left_dynamics import M as M_left
from kl_left_dynamics import c as c_left
from kl_left_dynamics import g as g_left
from kl_right_dynamics import M as M_right
from kl_right_dynamics import c as c_right
from kl_right_dynamics import g as g_right

from copy import deepcopy
from threading import RLock, Timer
from collections import deque

class MPC_control():

    def __init__(self, jt_num, appendage):

        self.lock = RLock()

        self.q_des = -10.0

        self.joint_idx =  jt_num       
        self.njts = 4

        self.appendage = appendage
        if appendage == "Right":
            self.M = M_right
            self.c = c_right
            self.joint = self.joint_idx
            if self.joint > 2: self.joint = self.joint - 1  # elbow removed, index back one spot
        elif appendage == "Left":
            self.M = M_left
            self.c = c_left
            self.joint = self.joint_idx - 5
            if self.joint > 7: self.joint = self.joint - 1  # elbow removed, index back one spot
        else:
            print "\n\n", "Please select appendage 'Right' or 'Left'", "\n\n"
            rospy.signal_shutdown("")

        self.rate = 300.0
        self.dt = 1.0/self.rate
        self.paps = 0.000145037738
        self.d2r = math.pi/180
        self.poff = 14.6959488
        self.goal_list1 = np.array([-20, -20, -20, -20]) * self.d2r
        self.goal_list2 = np.array([-60, 20, 20, 20]) * self.d2r

        self.pPlus_all = []   # generally, variables with 'Plus' refer to pressure, torque, coefficient, etc that correlate motion in the positive q direction
        self.pMinus_all = []  # variables with 'Minus' correlate with motion in the negative q direction
        self.q_all = []
        self.q_dot_all = []

        self.pPlus = 0
        self.pMinus = 0
        self.q = 0
        self.q_dot = 0
        self.q_ref = 0
        self.G = np.eye(self.njts)

        self.time = 0
        self.time_0 = None

        I = np.eye(self.njts)

        # tuning params ------------------------------------------
        self.Q = [10.0, 10.0, 5.0, 5.0]    # Error cost 
        self.R = 0.0                          # Velocity cost at end of horizon
        self.S = 0.00005                      # Stiffness cost
        self.V = 0.7 * np.eye(2)              # cost on changing pdes
        self.slew_rate = 2.5*self.dt          # contraint on difference between pdes at each time step
        self.ki = 0.005                       # integrator magnitude
        self.expscale = 30                    # scales integrator with q_dot
        self.Kd       = I * 1.229             # estimated damping coefficient in the joint
        self.Ks       = I * 12.2844           # estimated spring coefficient in the joint
        self.p_target = 15                    # target stiffness pressure
        # -------------------------------------------------------

        ############## these were found from pressure dynamics data
        self.gammaPlus  = I * 1.3512
        self.gammaMinus = I * 0.9465
        self.alphaPlus  = np.diag(np.array([1.2023, 1.3066, 2.8084, 3.5161]))
        self.alphaMinus = self.alphaPlus
        self.betaPlus   = np.diag(np.array([1.2015, 1.3183, 2.8043, 3.5069]))
        self.betaMinus  = self.betaPlus
        ############## these were found from pressure dynamics data

        # joint limits
        self.q_max = 120*self.d2r

        # All pressure values used in this MPC controller are in psig.
        # The pressure controller operates in psia, so all
        # published / subscribed pressure values are in psia.
        self.p_max = 25.3041
        self.p_min = 0.0
        self.u_prev = [self.p_target, self.p_target] 

        self.dintlast = 0.0
        self.q_ref = self.goal_list1[self.joint]
        self.theta = 0.0

        self.q_arr = deque([])
        self.q_ref_arr = deque([])
        self.time_arr = deque([])
        self.q_arr = deque([])
        self.pPlus_arr = deque([])
        self.pMinus_arr = deque([])
        self.pPlusd_arr = deque([])
        self.pMinusd_arr = deque([])
        self.q_dot_arr = deque([])
        self.dint_arr = deque([])
        self.p_target_arr = deque([])
        self.q_goal_arr = deque([])
        self.theta_arr = deque([])

        self.pdes_all = []
        self.pdes_all_ = np.ones(8)*14.7 

        publisher_name = '/Joint' + str(self.joint_idx) + 'DesiredPressures'

        rospy.Subscriber("/PressureData", sensor, self.pressure_callback, tcp_nodelay=True)
        rospy.Subscriber("/KLJointAngles",j_a_msg,self.angle_callback, tcp_nodelay = True)

        if appendage == "Right":
            rospy.Subscriber('/Joint0DesiredPressures', mpc_msg, self.pdes_callback0, tcp_nodelay=True)
            rospy.Subscriber('/Joint1DesiredPressures', mpc_msg, self.pdes_callback1, tcp_nodelay=True)
            rospy.Subscriber('/Joint3DesiredPressures', mpc_msg, self.pdes_callback3, tcp_nodelay=True)
            rospy.Subscriber('/Joint4DesiredPressures', mpc_msg, self.pdes_callback4, tcp_nodelay=True)
        else:
            rospy.Subscriber('/Joint5DesiredPressures', mpc_msg, self.pdes_callback5, tcp_nodelay=True)
            rospy.Subscriber('/Joint6DesiredPressures', mpc_msg, self.pdes_callback6, tcp_nodelay=True)
            rospy.Subscriber('/Joint8DesiredPressures', mpc_msg, self.pdes_callback8, tcp_nodelay=True)
            rospy.Subscriber('/Joint9DesiredPressures', mpc_msg, self.pdes_callback9, tcp_nodelay=True) 

        rospy.Subscriber("/SetJointGoal", j_a_msg, self.joint_goal_callback)

        self.pressure_pub = rospy.Publisher(publisher_name, mpc_msg, queue_size = 1, tcp_nodelay = True)

        print "Init cleared"

    def pressure_callback(self, data):
        self.lock.acquire()
        try:
            self.pPlus_all = []
            self.pMinus_all = []
            tj = 0

            for i in range(self.njts):
                self.pPlus_all.append( data.p[int(2*tj)]*self.paps - self.poff )
                self.pMinus_all.append( data.p[int(2*tj+1)]*self.paps - self.poff )
                
                # skip elbow
                if tj == 1: tj = tj + 2
                else: tj = tj + 1
        finally:
            self.lock.release()

    def angle_callback(self, data):
        self.lock.acquire()
        tj = 0
        self.q_all_ = []
        self.q_dot_all_ = []

        try:
            for i in range(0,len(data.joint_angles)):  # loops through appendages (right, left, hip)
                if data.joint_angles[i].appendage == self.appendage:
                    for j in range(self.njts):  # loop through joints in the appendage
                        
                        self.q_all_.append( data.joint_angles[i].q[tj] )
                        self.q_dot_all_.append( data.joint_angles[i].q_dot[tj] )

                        # skip elbow
                        if tj == 1: tj = tj + 2
                        else: tj = tj + 1
        finally:
            self.lock.release()

    def get_pdes_data(self, data, jt):
        self.lock.acquire()
        try:
            # arrange self.pdes_all_ for copying into input u vector
            # which has grav vals first, then pPlus vals, then pMinus vals
            i1 = jt
            i2 = jt + self.njts
            if data.flag == 1:
                self.pdes_all_[i1] = data.p_d[0] - self.poff
                self.pdes_all_[i2] = data.p_d[1] - self.poff
            else:
                self.pdes_all_[i1] = self.p_target 
                self.pdes_all_[i2] = self.p_target 
        finally:
            self.lock.release()

    def pdes_callback0(self, data):
        self.get_pdes_data(data, 0)

    def pdes_callback1(self, data):
        self.get_pdes_data(data, 1)

    def pdes_callback3(self, data):
        self.get_pdes_data(data, 2)

    def pdes_callback4(self, data):        
        self.get_pdes_data(data, 3)
        
    def pdes_callback5(self, data):
        self.get_pdes_data(data, 0)

    def pdes_callback6(self, data):
        self.get_pdes_data(data, 1)
        
    def pdes_callback8(self, data):
        self.get_pdes_data(data, 2)

    def pdes_callback9(self, data):
        self.get_pdes_data(data, 3)

    def joint_goal_callback(self, data):

        for i in range(0,len(data.joint_angles)):
                if data.joint_angles[i].appendage == self.appendage:
                    if data.joint_angles[i].flag != 1:
                        print "No Joint Data"
                        return                  
                    self.q_des = data.joint_angles[i].q[self.joint_idx]

    def get_data(self):
        self.lock.acquire()
        try:
            self.q_all = self.q_all_
            self.q_dot_all = self.q_dot_all_
            self.q = self.q_all[self.joint]
            self.q_dot = self.q_dot_all[self.joint]
            self.pPlus = self.pPlus_all[self.joint]
            self.pMinus = self.pMinus_all[self.joint]
            self.pdes_all = self.pdes_all_

            self.x = np.hstack([self.q_dot_all, self.q_all, self.pPlus_all, self.pMinus_all])
            self.x = np.matrix(self.x)
            self.x = np.transpose(self.x)

            t = time.time()
            
            if self.time_0 == None:
                self.time_0 = t
            self.timePrev = self.time
            self.time = t - self.time_0

        finally:
            self.lock.release()

    def control(self):
        if self.time > 500:
            rospy.signal_shutdown("")

        self.get_data()

        q_goal = self.get_q_goal()    

        dint = self.integrator(q_goal)

        A, B = self.build_model()
        Ad, Bd = self.discretize(A, B, 2)
        A, B, dstrb = self.build_disturbance_model(Ad, Bd)
        x0 = np.array([self.q_dot, self.q, self.pPlus, self.pMinus]).flatten('F').tolist()

        V = self.V.flatten('F').tolist()
        thetaPhi = self.getAdaptiveParams(q_goal)

        result = mpc_solver.runController(
                                           self.Q[self.joint],
                                           self.S,
                                           V,
                                           A,
                                           B,
                                           x0,
                                           self.u_prev,
                                           self.q_max,
                                           self.p_max,
                                           self.p_min,
                                           self.p_target,
                                           self.slew_rate,
                                           q_goal,
                                           dint,
                                           dstrb,
                                           -thetaPhi,
                                           2)

        pdesPlus = result[0][0]
        pdesMinus = result[1][0]

        u = [pdesPlus - thetaPhi + self.poff, pdesMinus - thetaPhi + self.poff]

        msg = mpc_msg(u, 1)

        self.pressure_pub.publish(msg)
        rospy.sleep(.00005)

        self.append_data(q_goal, u, dint)
        self.q_last = self.q
        self.u_prev = u

    def get_q_goal(self):
        period = 40
        if math.sin(2*math.pi*self.time/period) > 0:
            q_goal = self.goal_list1[self.joint]
        else:
            q_goal = self.goal_list2[self.joint]

        return q_goal

    def integrator(self, q_goal):
        interror = q_goal - self.q
        errord = self.q_dot       
        dintsum = self.ki*math.exp(-self.expscale*abs(errord))*interror + self.dintlast 
        limit = 1.0
        dint = min(limit, max(-limit, dintsum))
        self.dintlast = dint

        return dint

    def build_model(self):

        # tau = M*qdd + (C + Kd)*qd + Ks*q + G                                       - rigid robot arm model
        # tau = gammaPlus*pPlus - gammaMinus*pMinus                                  - pressure-to-torque model
        # pd = alpha*p + beta*pdes                                                   - first-order model for pressure dynamics
        # 
        # qdd = -Minv*((C+Kd)*qd - Ks*q + gammaPlus*pPlus - gammaMinus*pMinus - G)   - combined model

        z = np.zeros([self.njts, self.njts])
        I = np.eye(self.njts)

        M = np.matrix(self.M(dyn_params, self.q_all))
        M = M.reshape(self.njts, self.njts)
        cor = np.matrix(self.c(dyn_params, self.q_all, self.q_dot_all))
        cor = np.diag(cor)

        cKd = cor + self.Kd

        self.G = np.diag(g_right(dyn_params, self.q_all))

        Minv_cKd = linalg.solve(M, cKd)
        Minv_Ks = linalg.solve(M, self.Ks)
        Minv_gp = linalg.solve(M, self.gammaPlus)
        Minv_gm = linalg.solve(M, self.gammaMinus)
        Minv = linalg.inv(np.matrix(M))

        A_r1 = np.hstack([-Minv_cKd, -Minv_Ks, Minv_gp, -Minv_gm])
        A_r2 = np.hstack([I, z, z, z])
        A_r3 = np.hstack([z, z, -self.alphaPlus, z])
        A_r4 = np.hstack([z, z, z, -self.alphaMinus])
        A = np.vstack([A_r1, A_r2, A_r3, A_r4])

        B_r1 = np.hstack([-Minv, z, z])
        B_r2 = np.hstack([z, z, z])
        B_r3 = np.hstack([z, self.betaPlus, z])
        B_r4 = np.hstack([z, z, self.betaPlus])
        B = np.vstack([B_r1, B_r2, B_r3, B_r4])

        return A, B

    def discretize(self, A, B, mthd):
        
        if mthd == 1: # Matrix Exponential
            Ad = linalg.expm2(A*self.dt)
            eye = np.eye(4)
            Asinv = linalg.solve(A,eye)
            Bd = np.dot(np.dot(Asinv,(Ad-eye)),B)
        
        elif mthd == 2:  # Zero order hold, allows A to be singular, needed for model 3
            C = np.eye(self.njts*4)
            D = 0
            Ad, Bd, Cd, Dd, dt = spicycigs.cont2discrete((A, B, C, D),self.dt, method='bilinear')
        else:
            rospy.signal_shutdown('No discretization method selected')

        Ad = np.matrix(Ad)
        Bd = np.matrix(Bd)

        return Ad, Bd

    def build_disturbance_model(self, Ad, Bd):
        i = self.joint
        i2 = i + self.njts
        i3 = i + self.njts*2
        i4 = i + self.njts*3

        g_temp = np.transpose(np.matrix(np.diag(self.G)))
        pdes_temp = np.transpose(np.matrix(self.pdes_all))

        u_all = np.vstack([g_temp, pdes_temp])
        u_all = np.matrix(u_all)

        x1 = Ad*self.x + Bd*u_all

        dstrb1 = x1[i,0] - Ad[i,i]*self.x[i,0] - Ad[i,i2]*self.x[i2,0] - \
                           Ad[i,i3]*self.x[i3,0] - Ad[i,i4]*self.x[i4,0] - \
                           Bd[i,i2]*u_all[i2,0] - Bd[i,i3]*u_all[i3,0]
                         # Bd[i,i]*u_all[i] is gravity for link i, which is treated as a distrubance
        dstrb2 = x1[i2,0] - Ad[i2,i]*self.x[i,0] - Ad[i2,i2]*self.x[i2,0] - \
                            Ad[i2,i3]*self.x[i3,0] - Ad[i2,i4]*self.x[i4,0] - \
                            Bd[i2,i2]*u_all[i2,0] - Bd[i2,i3]*u_all[i3,0]
        dstrb = np.array([dstrb1, dstrb2, 0, 0]).flatten('F').tolist()

        A1 = np.hstack([Ad[i,i], Ad[i,i2], Ad[i,i3], Ad[i,i4]])
        A2 = np.hstack([Ad[i2,i], Ad[i2,i2], Ad[i2,i3], Ad[i2,i4]])
        A3 = np.hstack([Ad[i3,i], Ad[i3,i2], Ad[i3,i3], Ad[i3,i4]])
        A4 = np.hstack([Ad[i4,i], Ad[i4,i2], Ad[i4,i3], Ad[i4,i4]])
        A = np.vstack([A1, A2, A3, A4])
        A = A.flatten('F').tolist()

        B1 = np.hstack([Bd[i,i2], Bd[i,i3]])  # gravity is all in dstrb, so don't need it in B
        B2 = np.hstack([Bd[i2,i2], Bd[i2,i3]])
        B3 = np.hstack([Bd[i3,i2], Bd[i3,i3]])
        B4 = np.hstack([Bd[i4,i2], Bd[i4,i3]])
        B = np.vstack([B1, B2, B3, B4])
        B = B.flatten('F').tolist()

        return A, B, dstrb

    def getAdaptiveParams(self, q_goal):
        alpha = 1
        L = 5
        q_ref_dot = alpha*(q_goal - self.q_ref)
        q_err = self.q - self.q_ref
        Phi = self.q
        theta_dot = -L * Phi * q_err

        delTime = self.time - self.timePrev
        self.q_ref = self.q_ref + q_ref_dot * delTime
        self.theta = self.theta + theta_dot * delTime

        thetaPhi = self.theta * Phi

        return thetaPhi
    
    def append_data(self, q_goal, u, dint):
        self.q_arr.append(self.q)
        self.q_ref_arr.append(self.q_ref)
        self.q_dot_arr.append(self.q_dot)
        self.q_goal_arr.append(q_goal)
        self.time_arr.append(self.time)
        self.pPlus_arr.append(self.pPlus)
        self.pMinus_arr.append(self.pMinus)
        self.pPlusd_arr.append(u[0])
        self.pMinusd_arr.append(u[1])
        self.dint_arr.append(dint)
        self.p_target_arr.append(self.p_target)
        self.theta_arr.append(self.theta)

    def write_data(self):

        print "\n\n\nWriting data"

        savefile = os.environ['HOME'] + '/somepath/raw_data/mpc_data_j' + str(self.joint_idx)

        spicy.savemat(savefile,mdict={
                      'time':self.time_arr,
                      'q':self.q_arr,
                      'q_ref':self.q_ref_arr,
                      'q_dot':self.q_dot_arr,
                      'q_goal':self.q_goal_arr,
                      'pPlus':self.pPlus_arr,
                      'pMinus':self.pMinus_arr,
                      'pPlusd':self.pPlusd_arr,
                      'pMinusd':self.pMinusd_arr,
                      'rate_k':self.rate,
                      'dint':self.dint_arr,
                      'p_target':self.p_target_arr,
                      'theta':self.theta_arr})

        print "Written to File\n\n\n"
        msg = mpc_msg([15.0, 15.0], 0)
        self.pressure_pub.publish(msg)
        rospy.sleep(.1)

        self.pressure_pub.publish(msg)

if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('--jt_num', action="store", default=1, dest="jt_num", type="int") # pick joint from 0-9, except 2 and 7 (elbows)
    parser.add_option('--appendage',action="store",default="Right", dest="appendage",type="string")
    (options, args) = parser.parse_args()

    mpc = MPC_control(options.jt_num, options.appendage)

    rospy.init_node('mpc' + str(mpc.joint) + '_control', anonymous=True)
    rospy.on_shutdown(mpc.write_data)
    rate = rospy.Rate(mpc.rate)

    while not rospy.is_shutdown():
        mpc.control()
        rate.sleep()