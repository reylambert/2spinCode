#!/usr/bin/env python
# coding: utf-8

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general, tensor_basis
from quspin.tools.measurements import obs_vs_time
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
import cmath

def makeBasis(N, S1, S2):
    basis1 = spin_basis_general(N=N, S=S1)
    basis2 = spin_basis_general(N=N, S=S2)
    basis  = tensor_basis(basis1, basis2)
    return basis

def spinOps(h1, h2, theta, phi, basis):    
    mag1   = norm(h1)
    mag2   = norm(h2)
    
    zComp1 = mag1*np.cos(theta)               #static comp. for 1
    xComp1 = mag1*np.sin(theta)*np.cos(phi)
    yComp1 = mag1*np.sin(theta)*np.sin(phi)
        
    minus1 = [xComp1/2 - 1j*yComp1/2, 0]
    plus1  = [xComp1/2 + 1j*yComp1/2, 0]
    
    zComp2 = mag2*np.cos(theta)               #static comp. for 2
    xComp2 = mag2*np.sin(theta)*np.cos(phi)
    yComp2 = mag2*np.sin(theta)*np.sin(phi)
    
    minus2 = [xComp2/2 - 1j*yComp2/2, 0]
    plus2  = [xComp2/2 + 1j*yComp2/2, 0]
    
    static1 = [
        ["z|",[[zComp1, 0]]],   #z comp 1
        ["-|", [minus1]],       #- op 1
        ["+|", [plus1]],        #+ op 1
    ]
    
    static2 =  [
        ["|z",[[zComp2, 0]]],   #z comp 2
        ["|-", [minus2]],       #- op 2
        ["|+", [plus2]]         #+ op 2
    ]
    
    H1 = hamiltonian(static1, [], dtype=np.complex128, basis=basis) #to make operators for 1
    H2 = hamiltonian(static2, [], dtype=np.complex128, basis=basis) #to make operators for 2
    
    return H1, H2

def H_ini(h1, h2, J1, J2):      #initialize system in chosen direction, for initial state
    return -np.dot(h1, J1) - np.dot(h2, J2)

def H_dyn(h1, h2, L, V, J1, J2, Jzz):      #evolve initial state, H = -h1*J1 -h2*J2 + L*(Jzz)^2 + V*Jz_1*Jz_2
    return-np.dot(h1, J1) - np.dot(h2, J2) + L[0]*Jzz[0] + L[1]*Jzz[1] + V*J1[2]*J2[2]

def getJs(N, S1, S2, h1, h2):
    basis = makeBasis(N, S1, S2)
    
    Jx_1, Jx_2 = spinOps(h1, h2, np.pi/2, 0, basis)
    Jy_1, Jy_2 = spinOps(h1, h2, np.pi/2, np.pi/2, basis)
    Jz_1, Jz_2 = spinOps(h1, h2, 0, 0, basis)
    
    J1 = [Jx_1, Jy_1, Jz_1] 
    J2 = [Jx_2, Jy_2, Jz_2]
    
    Jzz = [J1[2]**2, J2[2]**2]
    
    return J1, J2, Jzz

def getObs(v_t, times, J1, J2):
    Obs_time = obs_vs_time(v_t, times, {'Jx_1':J1[0],
                                    'Jy_1':J1[1],
                                    'Jz_1':J1[2], 
                                    'Jx_2':J2[0],
                                    'Jy_2':J2[1],
                                    'Jz_2':J2[2]}, return_state=True)
    print(Obs_time.keys())

    Jx_1_time=Obs_time['Jx_1']
    Jy_1_time=Obs_time['Jy_1']
    Jz_1_time=Obs_time['Jz_1']

    Jx_2_time=Obs_time['Jx_2']
    Jy_2_time=Obs_time['Jy_2']
    Jz_2_time=Obs_time['Jz_2']

    J1_time = [Jx_1_time, Jy_1_time, Jz_1_time]
    J2_time = [Jx_2_time, Jy_2_time, Jz_2_time]
    
    return J1_time, J2_time

def takeZphi (z1, phi1, z2, phi2):
    h1 = [0,0,0]
    h2 = [0,0,0]
    
    h1[0] = np.sqrt(1-z1**2)*np.cos(phi1)
    h1[1] = np.sqrt(1-z1**2)*np.sin(phi1)
    h1[2] = z1
    
    h2[0] = np.sqrt(1-z2**2)*np.cos(phi2)
    h2[1] = np.sqrt(1-z2**2)*np.sin(phi2)
    h2[2] = z2
    
    return h1, h2

def makeZphi_h (h1, h2):
    z1   = h1[2]/norm(h1)
    phi1 = np.arctan2(h1[1], h1[0])
    
    z2   = h2[2]/norm(h2)
    phi2 = np.arctan2(h2[1], h2[0])
    
    z   = [z1, z2]
    phi = [phi1, phi2]
    
    return z, phi

def makeZphi_J(J1_time, J2_time):
    z1   = []
    phi1 = []
    
    z2   = []
    phi2 = []
    
    z1   = J1_time[2]/norm(J1_time) 
    phi1 = np.arctan2(J1_time[1].real, J1_time[0].real)
    
    z2   = J2_time[2]/norm(J2_time)
    phi2 = np.arctan2(J2_time[1].real, J2_time[0].real)
    
    return z1, phi1, z2, phi2

