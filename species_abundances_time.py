import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

EPS=0#10.**-4 #Threshold of immigration

S=2

meanA=-0.2

varA=0.1

r = 1
K = 1
d = 1

EXT = 0.03

temps1 = 1000

A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    

np.fill_diagonal(A,np.zeros(S))


#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=[v*1.0 for v in np.random.random(S)]  
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time=sol.t
    trajectories=sol.y
    return time, trajectories

time,trajectories = run(S)   

finalstate1 = [m for m in trajectories[:,-1]]
perturbation = np.random.normal(0.0,0.1,S)
perturbed_state = [abs(m+n) for m,n in zip(finalstate1, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=perturbed_state
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time2,trajectories2 = run(S)

t2 = [m+time[-1] for m in time2]

finalstate2 = [m for m in trajectories2[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state2 = [abs(m+n) for m,n in zip(finalstate2, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=perturbed_state2
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time3,trajectories3 = run(S)

t3 = [m+t2[-1] for m in time3]

#####################################################################################################
#####################################################################################################

finalstate3 = [m for m in trajectories3[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state3 = [abs(m+n) for m,n in zip(finalstate3, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=perturbed_state3
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time4,trajectories4 = run(S)

t4 = [m+t3[-1] for m in time4]

finalstate4 = [m for m in trajectories4[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state4 = [abs(m+n) for m,n in zip(finalstate4, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=perturbed_state4
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time5,trajectories5 = run(S)

t5 = [m+t4[-1] for m in time5]

#####################################################################################################
#####################################################################################################

finalstate5 = [m for m in trajectories5[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state5 = [abs(m+n) for m,n in zip(finalstate5, perturbation)]

#####################################################################################################
#####################################################################################################

def run(S, tmax=temps1, EPS=EPS,**kwargs ):

    def eqs(t,x):
        dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x)
        dx[x<=EPS]=0#np.maximum(0,dx[x<=EPS])  #if a species is below threshold, its abundance cannot keep decreasing
        x[x <= EPS] = 0   # Set abundance to zero if below threshold
        return dx

    x0=perturbed_state5
    sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
    time2=sol.t
    trajectories2=sol.y
    return time2, trajectories2    

time6,trajectories6 = run(S)

t6 = [m+t5[-1] for m in time6]

finalstate6 = [m for m in trajectories6[:,-1]]
perturbation = np.random.normal(0.0,1.5,S)
perturbed_state6 = [abs(m+n) for m,n in zip(finalstate6, perturbation)]

#####################################################################################################
#####################################################################################################

for i in range(len(trajectories)):
    plt.plot(time,trajectories[i]) 
for i in range(len(trajectories2)):
    plt.plot(t2, trajectories2[i])
for i in range(len(trajectories3)):
    plt.plot(t3, trajectories3[i])
for i in range(len(trajectories4)):
    plt.plot(t4, trajectories4[i])
for i in range(len(trajectories5)):
    plt.plot(t5, trajectories5[i])    
for i in range(len(trajectories6)):
    plt.plot(t6, trajectories6[i])  
    
plt.show()
