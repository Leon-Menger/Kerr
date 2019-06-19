#!/usr/bin/env python

'''
    Copyright (C) - All Rights reserved
    This file is meant solely for educational purpose and can be copied,
    modified and distributed while referencing the source (copy this header). 
    Any commercial use is forbidden unless specified otherwise by the author.
    Written by Léon Geiger <leonelvis2.0@gmail.com>, Juni 2019
'''

#libraries used up until now
import numpy as np                          #strong tool and math library
import math                                 #basic math routines
from matplotlib import pyplot as plt		#for creating plots
from mpl_toolkits.mplot3d import Axes3D		#for 3d plots
from matplotlib import cm                   #colormaps for our plot
from matplotlib.colors import LightSource   #for a shading effect in our 3D plot
import timeit                               #times the steps involved
from scipy.integrate import solve_ivp       #imports a convenient ODE solving method
import datetime


__author__ = "Léon Geiger"
__version__ = "1.2"
__status__ = "Production"


'''
    This programm heavily relies on the equations presented in:
    "ODYSSEY: A PUBLIC GPU-BASED CODE FOR GENERAL RELATIVISTIC
    RADIATIVE TRANSFER IN KERR SPACETIME"
        Hung-Yi Pu et al.
    
    The equation number X as stated in our summary of the paper 
    will henceforth be indicated by (X) in the comments.
    
    Link to the paper: https://arxiv.org/pdf/1601.02063.pdf
'''


def InitAndConserved(alpha, beta, r_obs, theta_obs, phi_obs, a):
    '''
    calculates the initial conditions for a given set of
    observer and object values
    Output format: [array(initial values), array(conserved quantities)]
        where: 
            initial values = [r, theta, phi, t, p_r, p_theta,\
                              r_dot, theta_dot, phi_dot, t_dot, p_r_dot, p_theta_dot]
                          
            conserved quantities = [Energy, Lz, kappa]
    '''
    
    #starts timer for this method
    INITstart = timeit.default_timer()
    
    #this bit avoid division by zero or numbers too small to handle but
    #keeps everything more than precise enough (--> just a numerical workaround)
    if abs(theta_obs) <= 10e-8:
        theta_obs = np.sign(theta_obs) * 10e-8
        if theta_obs==0: theta_obs = 10e-8
    if abs(phi_obs) <= 10e-8:
        phi_obs = np.sign(phi_obs) * 10e-8
        if phi_obs==0: phi_obs = 10e-8
    
    #calculates needed trig functions and a*a once to avoid
    #time expenses and numerical deviations
    sin_tobs, cos_tobs, sin_pobs, cos_pobs = np.sin(theta_obs), np.cos(theta_obs), np.sin(phi_obs), np.cos(phi_obs)
    a2 = a*a
    
    #calculates primed coordinates (11)&(12)
    D = math.sqrt(r_obs**2 + a2)*sin_tobs - beta * cos_tobs
    xprime = D * cos_pobs - alpha * sin_pobs
    yprime = D * sin_pobs + alpha * cos_pobs
    zprime = r_obs * cos_tobs + beta * sin_tobs
    
    #calculates initial values for r, theta, phi and t: (13)-(15)
    w = xprime*xprime + yprime*yprime + zprime*zprime- a2           #shortens r equation
    r = np.sqrt((w+math.sqrt(w*w + 4* a2 * zprime*zprime))/2)       #radius
    theta = np.arccos(zprime/r)                                     #theta
    phi = np.arctan2(yprime, xprime)                                #phi
    #sets start time to 0
    t = 0                                                           #time
    
    #as we use inverse RT we need the negative velocities
    zdot = -1
    
    #another batch of constants we need alot
    r2, sin_t, cos_t, sin_p, cos_p, cos_PHI = r*r, np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi), np.cos(phi - phi_obs)
    
    #miscellaneous
    R = math.sqrt(r2 + a2)
    Sigma = r2 + a2 * cos_t*cos_t
    Delta = r2 - 2.*r + a2
    
    r_dot = zdot * (- r*R*sin_t*sin_tobs*cos_PHI - R*R * cos_t * cos_tobs)/Sigma    #r velocity (16)
    theta_dot = zdot * (r*sin_t*cos_tobs - R*cos_t*sin_tobs*cos_PHI)/Sigma          #theta velocity (17)
    phi_dot = zdot * (sin_tobs * sin_p)/(R*sin_t)                                   #phi velocity (18)

    p_r = Sigma/Delta * r_dot                   #radius momentum (2)                         
    p_theta = Sigma * theta_dot                 #theta momentum (3)
    
    #squared and linear conserved energy (8)
    E2 = (Sigma - 2*r) * (r_dot*r_dot/Delta + theta_dot*theta_dot)+ Delta * phi_dot*phi_dot * sin_t*sin_t
    E = math.sqrt(E2)
    
    #conserved angular momentum and kappa (9)&(10)
    Lz = (((Sigma * Delta * phi_dot - 2.*a*r*E) * sin_t*sin_t)/(Sigma - 2*r))
    kappa = p_theta * p_theta + Lz*Lz * (1/sin_t)**2 + a2 * sin_t*sin_t * E2
    
    #calculates other initial values. Not all needed but convenient
    #in case of further use or future extended functionalities
    t_dot = E + (2*r*(r2 + a2)*E - 2*a*r*Lz)/(Sigma * Delta)                        #proper time change (5)
    p_r_dot = 1./(Sigma * Delta) * (-kappa*(r-1) + 2*r*(r2 + a2)*E2 - 2*a*E*Lz)\
              - (2*p_r*p_r * (r-1))/(Sigma)                                         #radius momentum change (6)
    p_theta_dot = (sin_t * cos_t)/(Sigma) * (Lz**2/sin_t**4 - a2 * E2)              #theta momentum change (7)
    
    #puts all 12 inital values in an array that can be called later
    #not directly defining the values as array elements supports readability
    init_values = [r, theta, phi, t, p_r, p_theta, r_dot, theta_dot, phi_dot, t_dot, p_r_dot, p_theta_dot]
    
    #array with conserved quantities in it
    #contains: Energy, angular momentum, kappa
    conserved = [E, Lz, kappa]
    
    #stops timer and prints the elapsed time
    INITstop = timeit.default_timer()
    #print('Time for Initial Values: {:f}'.format(INITstop-INITstart))
    #returns results as described in docstring
    return [init_values, conserved]
    

def GeodesicSolver(initial_parameters, conserved_quantities, a, IntMethod = "Radau"):
    '''
    Function to solve the geodesic equations. 
    Uses the solve_ivp function of the scipy library.
    The Integration method can be specified in the function parameters 
    and has "Radau" as default (very good for stiff problems).
    
    Takes initial_parameters and conserved_quantities formated as in InitAndConserved.
    Also takes the spin parameter and the integration method (default see above).
    
    Output format:
    returns ODE solution as "Bunch object" see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
	for details
    
    private functions in use:
        ToIntegrateIsToBe       --> This function will be integrated over with scipy
    '''
    
    #starts the timer for overall ODE calculation time
    ODEstart = timeit.default_timer()
    
    #convenient names for our conserved quantities
    E, Lz, kappa = conserved_quantities[0], conserved_quantities[1], conserved_quantities[2]
    

    def ToIntegrateIsToBe(time, y0):
        '''
        This is a local function that is integrated over using scipys
        solve_ivp to get the geodesics.
        In this function equations (2)-(7) are to be implemented.
        '''
        #container for new values
        dv = np.zeros(6)
        
        #convenient names
        r, theta, phi, t, p_r, p_theta = y0[0], y0[1], y0[2], y0[3], y0[4], y0[5]
        
        #numerical workaround for poles and division by zero
        if abs(theta) <= 10e-8:
            theta = np.sign(theta)*10e-8
            if theta==0: theta=10e-8
        if abs(phi) <= 10e-8:
            phi = np.sign(phi)*10e-8
            if phi==0: phi=10e-8
            
        #miscellaneous variables or further use
        Sigma = r*r + a*a * np.cos(theta)*np.cos(theta)
        Delta = r*r - 2*r + a*a
		
        #another numerical workaround (only needed at high precission)
        if abs(Delta) <= 1e-8:
            Delta = np.sign(Delta)*1e-8
            if Delta == 0: Delta = 1e-8
        if abs(Sigma) <= 1e-8:
            Sigma = np.sign(Sigma)*1e-8
            if Sigma == 0: Sigma = 1e-8
        
		#more convenient names, also we want to fix numerical values like sin&cos
        sin_t = np.sin(theta)
        r2, a2, E2 = r*r, a*a, E*E
		
		#for simplifying equation 5
        kappa  = p_theta * p_theta + Lz*Lz * (1/sin_t)**2 + a*a * sin_t**2 * E2
		
		#implementation of the differetial equations
		#Equations (3),(4),(9)-(12) from the paper by Pu et al.
		#negative for inverse raytracing
        dv[0] -= Delta/Sigma * p_r
        dv[1] -= 1/Sigma * p_theta
        dv[2] -= (2*a*r*E +(Sigma - 2*r) * Lz * (1/sin_t)**2)/(Sigma * Delta)
        dv[3] -= E + (2*r*(r2 + a2)*E-2*a*r*Lz)/(Sigma * Delta)
        dv[4] -= 1/(Delta * Sigma) * (-kappa*(r-1) + 2.0*r*(r2 + a2)*E*E - 2*a*E*Lz) - (2*p_r*p_r*(r-1))/(Sigma)
        dv[5] -= (sin_t * np.cos(theta))/(Sigma) * ((Lz*Lz)/(sin_t**4) - a2*E2)
        
        return np.array(dv)
    
	#function that checks if a ray hits the accretion disk
    def accretion(t,y): return y[1]-np.pi/2
	#tells the integrator to stop when the function passes zero
    accretion.terminal = True
    
	#checks if a ray is too far away
    def Lost(t,y): return 15+y[0]*np.sin(y[1])*np.cos(y[2])
    Lost.terminal = True
    
	#checks if a ray turns around too much
    def Turn(t,y): return 8*np.pi-y[2]
    Turn.terminal = True
   	
	#stops integration if ray hits the Event Horizon
    def HorizonHit(t,y): return abs(y[0])-(1 + np.sqrt(1 - a*a))
    HorizonHit.terminal = True
        
	#gets the solution via the solve_ivp method
	#see reference in function docstring for a detailed documentation
    solution = solve_ivp(ToIntegrateIsToBe,[1e-3, 45],initial_parameters[0:6],\
                         method = IntMethod,events = (accretion, Lost, Turn, HorizonHit),\
                         dense_output=False)
						 
	#prints the time each geodesic took to calculate, uncomment to use
    #print('Time(Geodesic_solver): {:f}'.format(timeit.default_timer()-ODEstart))
        
    return solution 
        
        
def main():
    '''
    main function where all important parameters are defined
    and from where all other methods are called
    
    parameters:
        spin parameter: a || Mass: M || impact parameters: alpha, beta ||
        observer distance: r_obs || observer angles: theta_obs, phi_obs
    
    global functions in use:
        InitAndConserved        --> get initial Values for the ODE
    
    private functions in use:
        None
    '''

    allstart = timeit.default_timer()       #starts time the main function took overall
    
    
    ##############      Constants           ##############
    a = .9                                  #spinparameter of the BH:  -1<a<1
                                            #sign of a dictates spin direction
    M = 1		                            #mass of the BH (always use 1)
    alpha = .5                              #defines initial x position on observer plane
    beta = .5                               #defines initial y position on observer plane
    r_obs = 20                              #observer distance from BH
    theta_obs = 105*np.pi/180               #camera theta angle in BH system
    phi_obs = 0*np.pi/180                   #camera phi   angle in BH system
    
    
    ##############      Calculations        ##############
    ranger = 30                                    #radius for the array matrix
    colormap = np.zeros((2*ranger+1, 2*ranger+1))  #creates color array 
    fig = plt.gcf()             #creates plot environment (figure)
    fig.show()                  #shows the figure (empty)
    fig.canvas.draw()           #draws a canvas (empty) on fig
    
    timesum = 0                 #container for sum of all
    mean = 0                    #for mean value of time per ray
    N = 0                       #number of rays calculated
	#starts two nested loops to calculate a ray for each colormap element
    for i in range(-ranger,ranger+1):
        for j in range(-ranger,ranger+1):
            N+=1													#updates number of rays done	
            raystart = timeit.default_timer()						#starts timer for ray operations
            INITS = InitAndConserved(i*alpha, j*beta,\				#gets init values for ray
								r_obs, theta_obs, phi_obs, a)		#notice how alpha and beta are modified
            initials = INITS[0]										#gets inital pos from INIT
            conserved = INITS[1]									#gets conserved quantities
            solutions = GeodesicSolver(initials, conserved, a)		#calculates ray trajectory
	    	#convenient new names
            r,theta,phi = solutions.y[0].astype(float),solutions.y[1].astype(float),solutions.y[2].astype(float)
			
	    	#This part calculates the color of the ray
	    	#if the last theta angle is close enough to pi/2 the ray gets a color
            if np.abs(solutions.y[1][-1]-np.pi/2) <= 1e-2 :
                #This one is more accurate for a temperature distribution
                raycol = 1/(solutions.y[0][-1])**(2/3)
                #This one just shows the principle (used for the GIF)
                #raycol = 1-solutions.y[0][-1]/max(solutions.y[0])
            else:
				#rays that dont hit the disc are black
                raycol = 0
           
            colormap[j+ranger][i+ranger] = raycol					#puts color to colormap
            timesum += timeit.default_timer()-raystart				#updates timesum
            mean = timesum/N										#calculates new mean time per ray
        plt.imshow(colormap, cmap = "gray")							#plots the updated colormap
        fig.canvas.draw()											#redraws the canvas
	#ordered terminal output of current status (percentage is NOT linear with time)
        print(f"   {(i+ranger)/(2*ranger+1) * 100:.2f} % , Timemean = {mean:.4f}, Rays: {N} of {(2*ranger+1)**2}", end="\r")
	#shows final colormap
    plt.imshow(colormap, cmap = "gray")
    fig.canvas.draw()
    
    allstop = timeit.default_timer()				#prints time the main function took overall
	
    print("Main function elapsed: {:.2f}".format(allstop-allstart))		#self explanatory terminal output
    print("Mean time per Ray: {:.4f} ".format(mean))					# ^^
    print("Showing ", (2*ranger+1)**2, " Rays")							# 
	
    plt.show()                                      #show the plot
    return


#calls main function upon running the programm
if __name__ == "__main__":
    main()
