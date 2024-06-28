import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
        
# CODE TO EXPLORE THE STATISTICS OF STATES UNDER COMP + COOP FOR THE GENERALIZED LOTKA-VOLTERRA MODEL (Barbier et al., 2018)
# G AGUADÉ-GORGORIÓ

####################### PARAMETERS AND SIMULATION VALUES ##############################

meanA=-0.2

varA=0.1

r = 1
K = 1
d = 0.0

reps=100

Sstart = 2
Smax = 100

points = Smax - Sstart

temps1 = 1000
temps2 = 50

EPS= 10.**-10 #Threshold of extinction in GLV, computational issue mostly
EXT=0 # 0.03 #Threshold of extinction

COMPARISON = 0.001

SURVIVAL =  0.001

##################################################################################

# Interaction range:

Slist = []

GLV_Frac_stable = np.zeros(points) # number of runs that end in stable behavior
GLV_Frac_ext = np.zeros(points) # fraction of extinct species
GLV_Frac_feas = np.zeros(points)
GLV_frac_neglambda = np.zeros(points)

SL_Frac_stable = np.zeros(points) # number of runs that end in stable behavior
SL_Frac_ext = np.zeros(points) # fraction of extinct species
SL_Frac_feas = np.zeros(points)
SL_frac_neglambda = np.zeros(points)

SL_Frac_stable_posd = np.zeros(points) # number of runs that end in stable behavior
SL_Frac_ext_posd = np.zeros(points) # fraction of extinct species
SL_Frac_feas_posd = np.zeros(points)
SL_posd_frac_neglambda = np.zeros(points)

SL_Frac_stable_EXT = np.zeros(points) # number of runs that end in stable behavior
SL_Frac_ext_EXT = np.zeros(points) # fraction of extinct species
SL_Frac_feas_EXT = np.zeros(points)
SL_EXT_frac_neglambda = np.zeros(points)

SL_Frac_stable_EXTposd = np.zeros(points) # number of runs that end in stable behavior
SL_Frac_ext_EXTposd = np.zeros(points) # fraction of extinct species
SL_Frac_feas_EXTposd = np.zeros(points)
SL_EXTposd_frac_neglambda = np.zeros(points)

for k in range(points):

    startclock = timer()
    
    S = int( Sstart + k*(Smax-Sstart)/points )
    
    num_species = S
    
    print("row: ", S," of ",Smax)
    
    
    

    
    
    ###################################################################
    # SL WITH EXTINCTIONS
    
    d = 0.0
    EXT = 0.07
    
    stable = 0
    extinct = 0
    feas = 0
    neglambda = 0
    
    for j in range(0,reps): #Perform experiment many times
    
        A=np.random.normal(meanA, varA, size=(S,S)) #COOP matrix, all interactions uniform between 0 and coop    
    
        np.fill_diagonal(A,np.zeros(S)) # SUBLINEAR MODEL WITH NO DEATH EFFECT
    
        
        def run(S, tmax=temps1, EPS=EPS,**kwargs):
            def eqs(t,x):
                dx = r*( x**(0.75) ) - (d*x) + x*np.dot(A,x) 
                dx[x <= EXT] = 0 #np.maximum(0,dx[x<=EPS])  #if a species is below threshold, EXTINCT
                x[x <= EXT] = 0   # Set abundance to zero if below threshold
                return dx
            #Generate random initial conditions, but summing Xinit:
            x0 = [v for v in np.random.random(S)]           
            sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
            time=sol.t
            trajectories=sol.y
            return time, trajectories
    
        # Simulation: write spp trajectories
        time,trajectories = run(S)
        finalstate = [m for m in trajectories[:,-1]]
                            
        #################### CYCLE CHECK: RUN MORE TIME AND OBSERVE IF STATE IS DIFFERENT #########################
            
        def run(S,tmax=temps2,EPS=EPS,**kwargs):
            def eqs(t,x):
                dx = r*(x**(0.75)) - (d*x) + x*np.dot(A,x) 
                dx[x <= EXT] = 0 #np.maximum(0,dx[x<=EPS])  #if a species is below threshold, EXTINCT
                x[x <= EXT] = 0   # Set abundance to zero if below threshold
                return dx
            #Solve the system of equations:
            x0 = finalstate           
            sol=solve_ivp(fun=eqs, t_span=(0,tmax),y0=x0)
            time=sol.t
            trajectories=sol.y
            return time, trajectories
    
        # Simulation: write spp trajectories
        timeplus,trajectoriesplus = run(S)
        finalstateplus = [m for m in trajectoriesplus[:,-1]] 
            
        ###########################################################################################################
            
        # DO WE SEE A NEW STATE, AN ALREADY-SEEN STATE, OR CYCLING/CHAOTIC BEHAVIOUR?
        diff = 0.0
        for spp in range(S):
            if abs ( finalstate[spp] - finalstateplus[spp] ) > COMPARISON:
                diff += 1
                break
        # print("final SL state is:")
        # print(finalstate)
        # COUNT EXTINCTIONS, STABLE OR NOT... TRICKY?                    
        
        # if there is a species going exponential, the system might get a bit crazy, forget? 
        # count only the extinct if all species are below exponential
        if not np.any(np.array(finalstate) > 1000):
            extinct += ( np.sum(np.array(finalstate) < SURVIVAL) ) / S
        
        #COUNT STABLE
        if diff == 0:
            stable += 1
            if np.all(np.array(finalstate) > SURVIVAL):
                feas +=1
            #check if negative eigenvalue
            xstar_N = np.array(finalstate)
            J = np.zeros((num_species,num_species))
            for fil in range(num_species):
                for col in range(num_species):
                    J[fil,col] = xstar_N[fil]*A[fil,col]
        
            #correct J of extinct species: only alive species can have positive growth?
            
            Jdiag = np.zeros(num_species)
            impacts = np.dot(A,xstar_N)
            for esp in range(num_species):
                if xstar_N[esp] > EXT:
                    Jdiag[esp] = r*0.75*(xstar_N[esp]**(-0.25)) - d + impacts[esp]
                else:
                    Jdiag[esp] =  - d +  impacts[esp]              
            
            np.fill_diagonal(J,Jdiag)
            eigenvalues, _ = np.linalg.eig(J)
            dom_lambda = np.real(eigenvalues[np.argmax(np.real(eigenvalues))])
            if dom_lambda < 0:
                neglambda +=1
            else:
                print("SL EXT a stationary state with positive eigenvalue?")
            
    
    SL_EXT_frac_neglambda[k] = neglambda / float(reps)
        
    SL_Frac_stable_EXT[k] = stable / float(reps)
        
    SL_Frac_ext_EXT[k] = extinct / float(reps)

    SL_Frac_feas_EXT[k] = feas / float(reps)    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    Slist.append(S)    

    k+=1
        
        
    endclock = timer()
    print("Line runtime", endclock - startclock)




























################## PLOTS ###############################


#combined_array = np.column_stack((np.array(Slist), np.array(GLV_Frac_feas),  np.array(SL_Frac_feas),np.array(SL_Frac_feas_posd), np.array(GLV_Frac_stable),np.array(SL_Frac_stable), np.array(SL_Frac_stable_posd), np.array(1-GLV_Frac_ext),np.array(1-SL_Frac_ext),np.array(1-SL_Frac_ext_posd)))

#np.savetxt('figure3.txt', combined_array, fmt='%.6f')

#Figure 1: number of states

fig,(ax1, ax2, ax3, ax4)= plt.subplots(1,4,figsize=(16,4))


# Panel 2: Plot the evolution of mean and std of overlap matrix values
#ax1.plot(Slist, GLV_Frac_feas, label='GLV', color = "cadetblue", alpha = 1, linewidth = 2)
#ax1.plot(Slist, SL_Frac_feas, label='SL',  color = "orangered",alpha = 1,linewidth = 2)
#ax1.plot(Slist, SL_Frac_feas_posd, label='SL pos d',  color = "orangered",linestyle = "--", alpha = 0.3,linewidth = 2)
ax1.plot(Slist, SL_Frac_feas_EXT, label='SL ext',  color = "green", alpha = 1,linewidth = 2)
#ax1.plot(Slist, SL_Frac_feas_EXTposd, label='SL ext+posd',  color = "purple", alpha = 0.7,linewidth = 2)
ax1.set_xlabel('S')
ax1.set_ylabel('F')
#ax1.legend()
ax1.set_title('Fraction of feasible simulations')
ax1.set_ylim(-0.05, 1.05)
#ax1.axvline(x=128, linestyle='--', color='k')  # Vertical dashed line at S = 128


# Panel 2: Plot the evolution of mean and std of overlap matrix values
#ax2.plot(Slist, GLV_Frac_stable, label='GLV',  color = "cadetblue",alpha = 1,linewidth = 2)
#ax2.plot(Slist, SL_Frac_stable, label='SL',  color = "orangered",alpha = 1,linewidth = 2)
#ax2.plot(Slist, SL_Frac_stable_posd, label='SL',  color = "orangered",linestyle = "--",alpha = 0.3,linewidth = 2)
ax2.plot(Slist, SL_Frac_stable_EXT, label='SL',  color = "green", alpha =1,linewidth = 2)
#ax2.plot(Slist, SL_Frac_stable_EXTposd, label='SL',  color = "purple", alpha = 0.7,linewidth = 2)
ax2.set_xlabel('S')
ax2.set_ylabel('extinct')
#ax2.legend()
ax2.set_title('Fraction of stable simulations')
ax2.set_ylim(-0.05, 1.05)
#ax2.axvline(x=128, linestyle='--', color='k')  # Vertical dashed line at S = 128

# Panel 3: Plot the evolution of mean and std of overlap matrix values
#ax3.plot(Slist, 1-GLV_Frac_ext, label='GLV',  color = "cadetblue",alpha = 1,linewidth = 2)
#ax3.plot(Slist, np.ones(len(Slist)), label='SL',  color = "orangered",alpha = 1,linewidth = 2)
#ax3.plot(Slist, 1-SL_Frac_ext_posd, label='SL',  color = "orangered",linestyle = "--",alpha = 0.3,linewidth = 2)
ax3.plot(Slist, 1-SL_Frac_ext_EXT, label='SL',  color = "green", alpha = 1,linewidth = 2)
#ax3.plot(Slist, 1-SL_Frac_ext_EXTposd, label='SL',  color = "purple", alpha = 0.7,linewidth = 2)
ax3.set_xlabel('S')
ax3.set_ylabel('Extinct')
#ax3.legend()
ax3.set_title('Fraction of surviving species')
ax3.set_ylim(0.45, 1.05)
#ax3.axvline(x=128, linestyle='--', color='k')  # Vertical dashed line at S = 128


# Panel 4 EIGENVALUES 
#ax4.plot(Slist, GLV_frac_neglambda, label='GLV',  color = "cadetblue",alpha = 1,linewidth = 2)
#ax4.plot(Slist, SL_frac_neglambda, label='SL',  color = "orangered",alpha = 1,linewidth = 2)
#ax4.plot(Slist, SL_posd_frac_neglambda, label='SL',  color = "orangered",linestyle = "--",alpha = 0.3,linewidth = 2)
ax4.plot(Slist, SL_EXT_frac_neglambda, label='SL',  color = "green", alpha = 1,linewidth = 2)
#ax4.plot(Slist, SL_EXTposd_frac_neglambda, label='SL',  color = "purple", alpha = 0.7,linewidth = 2)
ax4.set_xlabel('S')
ax4.set_ylabel('frac neg lambda')
#ax3.legend()
ax4.set_ylim(-0.05, 1.05)
ax4.set_title('Fraction of linearly stable states')

fig.tight_layout()
nom = f"SLext_moredeath.png"
plt.savefig(nom, format='png')
plt.close()

