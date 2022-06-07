import pandas as pd
import datetime
from datetime import date
import numpy as np
import os
import math
from dente import dente
import time

current_dir = os.getcwd()

def logicalArraySmaller(l1, l2):
    newl = []
    for i in range(0,len(l1)):
        if l1[i] < l2[i]:
            newl.append(1)
        else:
            newl.append(0)
    return newl

def logicalArraySE(l1,l2):
    newl = []
    for i in range(0,len(l1)):
        if l1[i] <= l2[i]:
            newl.append(1)
        else:
            newl.append(0)
    return newl

def main(xls):

    ''' Loading the data '''
    df = pd.read_excel(xls) #'TFM2.xls'
    n = len(df)
    numvar = len(df.columns)

    df1 = df.drop(index=0)
    df2 = df.drop(index=len(df)-1)
    df1.index = range(len(df)-1)

    rendement = (df1.subtract(df2))/df2
    T = len(rendement)
    numvar1 = len(rendement.columns)
    

    ''' Various quantity initializations '''
    PSO_init = 2
    nrun = 2
    maquanti = [150, 100]

    ''' PSP PARAMETERS'''
    a = 0.5     # parametro funzione di rischio
    b = 2       # risk function parameter
    kd = 11     # numero minimo titoli in portafoglio
    ku = 30     #maximum number of holdings in the portfolio
    media = rendement.mean()    #vector of average returns

    gra_F_RHO = True        #fitness and risk function charts for the best portfolio of each run
    gra_diametro = True     #graphs of the diameters of the set of particles of each run
    gra_F_RHO_din_stat = False  # dynamic vs static fitness and risk function graphs
    gra_F_RHO_perc = False      #dynamic vs static fitness / RHO percentages graphs
    gra_FE = False          #efficient border chart

    c1 = 1.49618
    c2 = 1.49618
    w = 0.7298
    chi = 1
    a = chi*w
    W = chi*((c1*0.5)+(c2*0.5))
    valore_max_W = 2*(a+1)

    today = date.today()
    now = datetime.datetime.now()

    variable = 2*numvar
    P = variable

    tab1 = np.zeros([10,2])
    tab2 = np.zeros([numvar,6])


    ''' Portfolio initializations'''
    pi_v = np.linspace(0.0001,0.00505578351944675,5)
    pi_v = 0.0000967 #why do that???
    npi = 1 #len(pi_v) idk for this

    xfe = np.zeros([npi, 1])
    yfe = np.zeros([npi, 1])
    xfe_am = -1*np.ones([npi,1])
    yfe_am = -1*np.ones([npi,1])
    prtf_pi2 = np.zeros([2*numvar+11,npi])
    prtf_pi2 = np.zeros([2*numvar+11,npi])

    discrezione = 0/100

    #start computations that will be measured later
    ''' Iterations per pi '''
    for hij in range (0,npi):
        pi = pi_v
        am = True
        rho_pre = np.zeros([1,2])
        rend_pre = np.zeros([1,2])
        diam_prepost = np.zeros([2,2])

        for giro in range (0,1):
            

            epsilon = 1.0e-004  # penalizzazione violazione vincoli
            epsilon_or = epsilon    # pro benchmark
            percent_1 = 0.05    #quantity in (0,1) used to possibly update penalty parameters
            percent_2 = 0.90    # quantity in (0,1) used to possibly update penalty parameters
            Delta_k = 40    # number of iterations between two successive updates of penalty parameters
            Delta_k_rho = 20    # number of iterations between two successive updates of penalty parameter EPSILON
            max_weight = 10000  # maximum value of any penalty parameter
            min_weight = 0.0001 # minimum value of any penalty parameter
            max_epsilon = 1     #maximum value allowed for "epsilon"
            min_epsilon = 1.0e-15   #minimum value allowed for "epsilon"

            niter = maquanti[giro]

            ''' other initializations '''
            nomone = 'CdTFP' #id file
            nomonef = nomone
            nomone = nomone + '-' + str(giro) + '-' + str(pi) 
            nomoneD = nomone
            nomoneFIT = nomone
            nomoneRHO = nomone
            nomoneEPSILON = nomone
            nomonePESI = nomone
            nomoneTAB1 = nomone
            nomoneTAB2 = nomone
            nomonePRTF = nomone
            nomone = nomone + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '.txt'
            nomoneD = nomoneD + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-DIAMETRI.txt'
            nomoneFIT = nomoneFIT + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-FITNESS.txt'
            nomoneRHO = nomoneRHO + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-RHO.txt'
            nomoneEPSILON = nomoneEPSILON + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-EPSILON.txt'
            nomonePESI = nomonePESI + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-PESI.txt'
            nomoneTAB1 = nomoneTAB1 + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-CONFRONTI_DATI-pi_' + str(pi) + '.txt'
            nomoneTAB2 = nomoneTAB2 + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-CONFRONTI_PORT-pi_' + str(pi) + '.txt'
            nomonePRTF = nomonePRTF + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1] + '-CONFRONTI_TUTTO-pi_' + str(pi) + '.txt'
            nomonef = nomonef + '-' + str(today).replace('-','') + '-' + str(now)[-6:-1]

            ''' USED TO CREATE DIRECTORY TO ADD CALCULATED VALUES
            if (hij == 0) and (giro == 0):
                os.mkdir(nomonef)
            '''

            ''' CREATES THE DIFFERENT .txt FILES TO (POTENTIALLY) STORE DATA
            pathmio = nomonef + '\\'
            #missing lines 152 - 158 -> don't understand why they are here bc we never created those directories (ask for help)
            fid1 = open(nomone,'a') # per salvare i dati del migliore portafoglio di ogni run
            fid2 = open(nomoneD,'a') # per salvare i diametri di ogni i run
            fid3 = open(nomoneFIT,'a') # per salvare le fitness di ogni i run
            fid4 = open(nomoneRHO,'a') # per salvare i rho di ogni i run
            fid5 = open(nomoneEPSILON,'a') # per salvare gli pesi di ogni i run
            fid6 = open(nomonePESI,'a') # per salvare i pesi di ogni i run     
            '''

            step_conta = 1
            n_confro = math.floor(niter/step_conta)

            violazioni = np.zeros([nrun,1])

            diametro = np.zeros([1, niter])
            diametro_or = np.zeros([1, niter])
            conta_fitness = 0
            conta_RHO = 0
            conta_fitness_ug = 0
            conta_RHO_ug = 0

            if n_confro == niter/step_conta:
                conta_mat = np.zeros([n_confro,4])
            else:
                conta_mat = np.zeros([n_confro+1,4])

            if giro == 0:
                f_super_king = 1.0e+15
            

            ''' PSO algorithm'''
            for giacomo in range (0,nrun):
                epsilon = epsilon_or

                print('******************************************************************************************* ')
                print('******************************************************************************************* ')
                print('*** File identifier                : ', nomone)
                print('*** Current expected return value: ', pi)
                print('*** round                                : ', giro+1)
                if giro == 0:
                    print('*** PSO_init                            : ', PSO_init)            
                else:
                    print('*** PSO_init                            : ', 'Best population of round no. 1')
                print('*** Total number of iterations         : ', niter)
                print('*** Number of the current run             : ', giacomo+1)
                print('******************************************************************************************* ')
                print('******************************************************************************************* \n ')
            
                vmax_x = np.zeros([1,numvar])
                vmax_x_or = np.zeros([1,numvar])

                Delta_viol = np.zeros([P,1]) # overall constraints violation x constraints weights
                Delta_viol_OLD = np.zeros([P,1]) # overall OLD constraints violation x OLD constraints weights
                Delta_viol_b = 0.0 # best (among particles) overall constraints violation x constraints weights
                Delta_viol_b_OLD = 0.0 # best (among particles) overall OLD constraints violation x OLD constraints weights

                Delta_viol_or = np.zeros([P,1]) # pro benchmark
                Delta_viol_OLD_or = np.zeros([P,1]) # pro benchmark
                Delta_viol_b_or = 0.0 # pro benchmark
                Delta_viol_b_OLD_or = 0.0 # pro benchmark
            
                converg = np.zeros([niter,1]) # for storing risk function
                RR = np.zeros([niter,1]) # for storing risk function
                DD = np.zeros([niter,1]) # for storing constraint violations
                e_v = np.zeros([niter, 1]) # for dynamic epsilon storage
                v_v = np.zeros([niter, 8]) # for dynamic constraint storage
                p_v = np.ones([niter, 8]) # for storing dynamic weights
            
                converg_or = np.zeros([niter,1]) # for storing risk function
                RR_or = np.zeros([niter,1]) # pro benchmark
                DD_or = np.zeros([niter,1]) # for storing constraint violations

                # Inizializzazione y e vy
                y = np.zeros([P,variable])
                vy = np.zeros([P,variable])

                y_or = np.zeros([P,variable]) # pro benchmark
                vy_or = np.zeros([P,variable]) # pro benchmark


                ''' initialization of various functions, constraints and weights for constraints '''
                # !!! what are lines 257 to 264?? is pesi_vinc a function?
                pesi_vinc = [0, 0.41, 0.42, 0.52, 0.66, 0.46, 0.72, 0.22, 0.303]

                rho = np.zeros([P, 1]) # risk function
                R = np.zeros([P, 1]) # portfolio return
                vinc_1 = np.zeros([P, 1]) # budget constraint
                vinc_2 = np.zeros([P, 1]) # profitability constraint
                vinc_4 = np.zeros([P, 1]) # lower cardinality constraint
                vinc_5 = np.zeros([P, 1]) # upper cardinality constraint
                app_3 = np.zeros([P, numvar])
                vinc_7 = np.zeros([P, 1]) # constraint minimum fraction
                app_4 = np.zeros([P, numvar])
                vinc_8 = np.zeros([P, 1]) # constraint maximum fraction
                difference = np.zeros([T, numvar1])

                rho_or = np.zeros([P, 1]) 
                R_or = np.zeros([P, 1]) 
                vinc_1_or = np.zeros([P, 1]) 
                vinc_2_or = np.zeros([P, 1]) 
                vinc_4_or = np.zeros([P, 1]) 
                vinc_5_or = np.zeros([P, 1]) 
                app_3_or = np.zeros([P, numvar])
                vinc_7_or = np.zeros([P, 1]) 
                app_4_or = np.zeros([P, numvar])
                vinc_8_or = np.zeros([P, 1]) 
                difference_or = np.zeros([T, numvar1])


                ''' initialization constraint violation previous iteration '''
                vinc_1_OLD = np.zeros([P, 1]) # budget constraint
                vinc_2_OLD = np.zeros([P, 1]) # benchmark profitability constraint
                vinc_4_OLD = np.zeros([P, 1]) # benchmark cardinality constraint
                vinc_5_OLD = np.zeros([P, 1]) # benchmark cardinality constraint
                vinc_7_OLD = np.zeros([P, 1]) # constraint minimum fraction
                vinc_8_OLD = np.zeros([P, 1]) # constraint maximum fraction

                vinc_1_OLD_or = np.zeros([P, 1]) # pro benchmark
                vinc_2_OLD_or = np.zeros([P, 1]) # pro benchmark
                vinc_4_OLD_or = np.zeros([P, 1]) # pro benchmark
                vinc_5_OLD_or = np.zeros([P, 1]) # pro benchmark
                vinc_7_OLD_or = np.zeros([P, 1]) # pro benchmark
                vinc_8_OLD_or = np.zeros([P, 1]) # pro benchmark

                ''' Initialization of PSO'''
                if giro == 0:
                    if PSO_init == 1: #alternative PSO
                        k = 1
                        lambda1 = ((-(W-a-1))+(math.sqrt((W-a-1)^2-4*a)))/2
                        lambda2 = ((-(W-a-1))-(math.sqrt((W-a-1)^2-4*a)))/2
                        gamma1 = ((lambda1^k)*(a-lambda2)-(lambda2^k)*(a-lambda1))/(lambda1-lambda2)
                        gamma2 = (W*((lambda1^k)-(lambda2^k)))/(lambda1-lambda2)
                        sigma1 = gamma1^2
                        sigma2 = -gamma1*gamma2
                        sigma3 = gamma2^2
                        mu1 = ((sigma1+sigma3)+math.sqrt((sigma1+sigma3)^2-4*(sigma1*sigma3-sigma2^2)))/2
                        mu2 = ((sigma1+sigma3)-math.sqrt((sigma1+sigma3)^2-4*(sigma1*sigma3-sigma2^2)))/2
                        U = np.zeros([P,variable])
                        V = np.zeros([P,variable])
                        coeff1 = -((sigma3-mu1)/sigma2)
                        coeff2 = -((sigma3-mu2)/sigma2)

                        for i in range(0,variable):
                            U[i,i] = coeff1
                            U[variable+i,i] = 1
                            V[i,i] = coeff2
                            V[variable+i,i] = 1
                        
                        for p in range(0,variable):
                            vy[p,0:variable] = np.transpose(U[0:variable,p])
                            y[p,0:variable] = np.transpose(U[(variable+1):(2*variable),p])
                        
                        x = y[:, 1:numvar]
                        #print('test1', x.shape) for debugging
                        vx = vy[:, 1:numvar]
                    else: #standard PSO
                        x = np.random.rand(P, numvar)
                        #print('test2', x.shape) for debugging
                        vx = np.random.rand(P, numvar)
                else:
                    continue

                x_or = x
                vx_or = vx

                '''Optimization'''
                f = np.ones([P, 1])*1.0e+015 # fitness vector in each iteration
                f_king = 1.0e+015 # minor fitness found

                f_or = np.ones([P, 1]) * 1.0e+015 # fitness benchmark initialization
                f_king_or = 1.0e+015 # lower fitness my benchmark
                    
                # d = np.ones (1, numvar) * (1 / kd) # minimum fraction as per article
                # u = np.ones (1, numvar) * (1 / ku) # maximum fraction as per article
                # d = zeros (1, numvar) # minimum fraction
                # u = np.ones (1, numvar) # maximum fraction
                d = (0.000 / 100) * np.ones([1, numvar]) # minimum fraction
                u = (100/100) * np.ones([1, numvar]) # maximum fraction
                ddd = d[0][0]
                uuu = u[0][0]
                if (ddd.all() == 0): # automatic tolerance desetting
                    discretion = 0
                
                pb_x = np.concatenate((x, f), axis=1) # pbest: best position vector for particle - last column = value of the objective function
                #print(pb_x.shape)

                pb_rho = np.zeros([P, 1])  # pbest rho: vector rho associated with the best positions per particle
                pb_vio = np.zeros([P, 1])  # pbest violations: rho vector associated with the best position per particle
                pb_e = 0.0  # pbest epsilo: epsilon vector associated with the best positions per particle
                pb_vinc = np.zeros([1,8])  # pbest constraints: weights matrix associated with the best positions per particle
                pb_vinc_OLD = np.zeros([1,8])  # pbest constraints: weights matrix associated with the best positions per particle
                pb_pesi = np.zeros([1,8])  # pbest weights: weight matrix associated with the best positions per particle
                rho_b = 0  # fitness function best particle (b = best)
                rho_b_OLD = 0  # fitness function best previous particle (b = best)
            
                pb_x_or = x_or + f_or  # pbest benchmark
                pb_rho_or = np.zeros([P, 1])  # pbest rho: vector rho associated with the best positions per particle
                pb_vio_or = np.zeros([P, 1])  # pbest violations: rho vector associated with the best position per particle
                rho_b_or = 0  # best particle benchmark fitness function
                rho_b_OLD = 0  # fitness benchmark best previous particle function

                g_x = np.zeros([1, numvar + 1])  # gbest and its value of the objective function
            
                g_x_or = np.zeros([1, numvar + 1])  # gbest benchmark and relative value of the objective function

                temp_rendement = np.array(rendement.values.tolist())
                for t in range(0, T):
                    difference[t,:] = temp_rendement[t,:] - media    #difference and rendement is a 117x40 array
                    difference_or[t,:] = temp_rendement[t,:] - media

                prodotto_somma = np.zeros([P, T])
                prodotto_somma_or = np.zeros([P, T])


                for k in range(0,niter):
                    # 1) range calculation of maximum speed
                    for i in range(0,numvar): #numvar = 40
                        vmax_x[0][i] = abs(np.max(x[:,i]) - np.min(x[:,i]))
                        vmax_x_or[0][i] = abs(np.max(x_or[:,i]) - np.min(x_or[:,i]))
                    
                    
                    for p in range(0,P): #80
                        for i in range(0,numvar):
                            app_3[p, i] = max(0, d[0][i].all()-x[p,i])
                            app_4[p, i] = max(0, x[p,i]-u[0][i].all())

                            app_3_or[p, i] = max(0, d[0][i].all()-x_or[p,i])
                            app_4_or[p, i] = max(0, x_or[p,i]-u[0][i].all())

                        for t in range(0,T): #117
                            #this crashes because the operation on the right returns a list and not a value
                            #proabbly because the transpose is not changing anything
                            
                            prodotto_somma[p, t] = np.dot(difference[t,:], np.transpose(x[p,:]))
                            prodotto_somma_or[p, t] = np.dot(difference_or[t,:],np.transpose(x_or[p,:]))
                #print('prodotto_somma =', prodotto_somma)
                #print('prodotto_somma_or =', prodotto_somma_or)

                        R[p] = np.dot(x[p,:],np.transpose(media)) 
                        R_or[p] = np.dot(x_or[p,:],np.transpose(media))

                        rho[p] = np.dot((a/T),max(0,sum(prodotto_somma[p,:]))) + np.dot((1-a)*(1/T**(1/b)),max(0,-sum(prodotto_somma[p,:]))**b)**(1/b) - R[p] 
                        rho_or[p] = np.dot((a/T),max(0,sum(prodotto_somma_or[p,:]))) + np.dot((1-a)*(1/T**(1/b)),max(0,-sum(prodotto_somma_or[p,:]))**b)**(1/b) - R_or[p] 
                    
                        #line 443 - 447 -> pointless? calculations but never used later in the code


                        #2.7.1 calculations
                        vinc_1[p] = abs(sum(x[p,:])-1)
                        vinc_2[p] = max(0, np.dot(pi-x[p,:],np.transpose(media)))
                        vinc_4[p] = max(0,kd-sum(dente(x[p,:],ddd,uuu)))  #dente is an extenernal function TBD
                        vinc_5[p] = max(0,sum(dente(x[p,:],ddd,uuu))-ku)
                        vinc_7[p] = sum(app_3[p,:])
                        vinc_8[p] = sum(app_4[p,:]) 

                        vinc_1_or[p] = abs(sum(x_or[p,:])-1)
                        vinc_2_or[p] = max(0, np.dot(pi-x_or[p,:],np.transpose(media)))
                        vinc_4_or[p] = max(0,kd-sum(dente(x_or[p,:],ddd,uuu)))  #dente is an extenernal function TBD
                        vinc_5_or[p] = max(0,sum(dente(x_or[p,:],ddd,uuu))-ku)
                        vinc_7_or[p] = sum(app_3_or[p,:])
                        vinc_8_or[p] = sum(app_4_or[p,:])

                        #lines 465 - 477 useless like earlier, calculations but never actually used

                        #2.c.2
                        Delta_viol_OLD[p] = Delta_viol[p]
                        Delta_viol[p] = pesi_vinc[1]*vinc_1[p] + pesi_vinc[2]*vinc_2[p] + pesi_vinc[4]*vinc_4[p] + + pesi_vinc[5]*vinc_5[p]+ pesi_vinc[7]*vinc_7[p]+ pesi_vinc[8]*vinc_8[p]
                        # trace_Delta[p,k] = Delta_viol[p]  #useless for the moment
                        Delta_viol_OLD_or[p] = Delta_viol_or[p]
                        Delta_viol_or[p] = vinc_1_or[p]+vinc_2_or[p]+vinc_4_or[p]+vinc_5_or[p]+vinc_7_or[p]+vinc_8_or[p]

                    #2.d fitness function
                    f = rho + (1/epsilon)*Delta_viol
                    f_or = rho_or + (1/epsilon_or)*Delta_viol_or

                    for p in range(0,P):
                        if f[p] < pb_x[p, numvar]:
                            pb_x[p, numvar] = f[p]
                            if f[p] < f_king:
                                pb_rho[p] = rho[p]
                                pb_vio[p] = Delta_viol[p]
                                pb_e = epsilon
                
                                pb_vinc_OLD[0][0] = pb_vinc[0][0]
                                pb_vinc_OLD[0][1] = pb_vinc[0][1]
                                pb_vinc_OLD[0][3] = pb_vinc[0][3]
                                pb_vinc_OLD[0][4] = pb_vinc[0][4]
                                pb_vinc_OLD[0][6] = pb_vinc[0][6]
                                pb_vinc_OLD[0][7] = pb_vinc[0][7]

                                pb_vinc[0][0] = vinc_1[p]
                                pb_vinc[0][1] = vinc_2[p]
                                pb_vinc[0][3] = vinc_4[p]
                                pb_vinc[0][4] = vinc_5[p]
                                pb_vinc[0][6] = vinc_7[p]
                                pb_vinc[0][7] = vinc_8[p]

                                pb_pesi = pesi_vinc
                                f_king = f[p]

                                if giro == 0:
                                    ARABA0 = np.transpose(x)

                            rho_b_OLD = rho_b
                            rho_b = rho[p]
                            Delta_viol_b_OLD = Delta_viol_b
                            Delta_viol_b = Delta_viol[p]

                            for i in range(0, numvar):
                                pb_x[p,i] = x[p,i]

                        if giro == 0:
                            ARABA_ser = ARABA0
                            ARABA_z_n = 0*ARABA_ser[:,1]
                        
                        if f_or[p] < pb_x_or[p, numvar-1]:
                            pb_x_or[p, numvar-1] = f_or[p]
                            if f_or[p] < f_king_or:
                                pb_rho_or[p] = rho_or[p]
                                pb_vio_or[p] = Delta_viol_or[p]
                                f_king_or = f_or[p]
                            
                            rho_b_OLD_or = rho_b_or
                            rho_b_or = rho_or[p]
                            Delta_viol_b_OLD_or = Delta_viol_b_or
                            Delta_viol_b_or = Delta_viol_or[p]

                            for i in range(0, numvar):
                                pb_x_or[p,i] = x_or[p,i]
                    
                    ''' Identification of the best fitness and location of the particle '''
                    minimo = min(pb_x[0:,numvar])
                    posizione = np.where(pb_x[0:,numvar] == minimo)[0][0] #not too sure about this
                    RR[k] = pb_rho[posizione]
                    DD[k] = pb_vio[posizione]
                    e_v[k] = pb_e
                    v_v[k,:] = pb_vinc[0]
                    p_v[k,:] = pb_pesi[0]

                    minimo_or = min(pb_x_or[0:,numvar-1])
                    posizione_or = np.where(pb_x_or[0:,numvar-1] == minimo_or)[0][0] #not too sure about this
                    RR_or[k] = pb_rho_or[posizione_or]
                    DD_or[k] = pb_vio_or[posizione_or]

                    if k % 500 == 0:
                        continue #print statements to be added 

                    if k % Delta_k_rho == 0:
                        if rho_b >= rho_b_OLD:
                            epsilon = min(3*epsilon, max_epsilon)
                        elif rho_b < rho_b_OLD*percent_2:
                            epsilon = max(0.6*epsilon, min_epsilon)

                    g_x[0][numvar] = minimo    
                    g_x_or[0][numvar] = minimo_or

                    for i in range(0, numvar):
                        g_x[0][i] = pb_x[posizione, i]
                        g_x_or[0][i] = pb_x_or[posizione, i]

                    for p in range(0,P):
                        for i in range(0,numvar):
                            if (PSO_init == 1) or (PSO_init == 2):
                                r1 = 1.0
                                r2 = 1.0
                            else:
                                r1 = np.random.rand()
                                r2 = np.random.rand()
                                print(r1, r2)
                            
                            r1_or = r1
                            r2_or = r2

                            vx[p,i] = w*vx[p,i]+c1*r1*(pb_x[p,i]-x[p,i])+c2*r2*(g_x[0,i]-x[p,i])
                
                            if vx[p,i] > vmax_x[0,i]:
                                vx[p,i] = vmax_x[0,i]
                            x[p,i] = x[p,i] + vx[p,i]

                            vx_or[p,i] = w*vx_or[p,i]+c1*r1_or*(pb_x_or[p,i]-x_or[p,i])+c2*r2_or*(g_x_or[0,i]-x_or[p,i])
                            if vx_or[p,i] > vmax_x_or[0,i]:
                                vx_or[p,i] = vmax_x_or[0,i]
                            x_or[p,i] = x_or[p,i] + vx_or[p,i]
                    
                    ''' calculation of the diameter of the particles '''
                    for kk in range(0, P-1):
                        for jj in range(kk, P):
                            diametro[0, k] = max(diametro[0, k], np.linalg.norm(x[kk,:]-x[jj,:]) ,2)
                            diametro_or[0, k] = max(diametro_or[0, k], np.linalg.norm(x_or[kk,:]-x_or[jj,:]) ,2)
            
                    ''' storage of various quantities '''
                    converg[k,:] = g_x[:,len(g_x)]                  #not sure about this
                    converg_or[k,:] = g_x_or[:,len(g_x_or)]
                
                
                xxx = []
                for i in range(1, niter+1):
                    xxx.append(i)

                if gra_F_RHO:
                    altezza = max(converg[5], converg_or[5])
                    altezzo = min(min(converg), min(converg_or))
                    if niter < 5:
                        altezza = 1e3
                
                if gra_diametro:
                    uni = np.ones([niter,1])
                    altezza = max(diametro + diametro_or)*1.05

                ''' saving results for each single run '''
                g_z = dente(g_x[0][:-1], ddd, uuu)
                portfolio = np.transpose(g_x[0][:-1]) + np.transpose(g_z)
#CHANGED GIACOMO FROM 1 TO 0 TO TEST
                if giacomo == 0:        
                    convergenzaglobal = converg
                    RRglobal = RR
                    fit = g_x[0][len(g_x)]
                    rend_portaf = np.dot(g_x[0][:-1],np.transpose(media))
                    portfolioglobal = np.transpose(g_x[0][:-1])

                    convergenzaglobal_or = converg_or
                    RRglobal_or = RR_or
                    fit_or = g_x_or[0][len(g_x_or)]
                    rend_portaf_or = np.dot(g_x_or[0][:-1],np.transpose(media))
                    portfolioglobal_or = np.transpose(g_x_or[0][:-1])
                else:
                    continue
                    ''' section here is useless since we didnt declare the varibales yet 
                    convergenzaglobal = convergenzaglobal + converg
                    RRglobal = RRglobal + RR
                    fit = fit + g_x[0][len(g_x_or)]
                    rend_portaf = rend_portaf + np.dot(g_x[0][:-1],np.transpose(media))
                    portfolioglobal = portfolioglobal + np.transpose(g_x[0][:-1])

                    convergenzaglobal_or = convergenzaglobal_or + converg_or
                    RRglobal_or = RRglobal_or + RR_or
                    fit_or = fit_or + g_x_or[0][len(g_x_or)]
                    rend_portaf_or = rend_portaf_or + np.dot(g_x_or[0][:-1],np.transpose(media))
                    portfolioglobal_or = portfolioglobal_or + np.transpose(g_x_or[0][:-1]) 
                    '''
                
                conta_fitness = conta_fitness + sum(logicalArraySmaller(converg, converg_or))/niter
                conta_RHO = conta_RHO + sum(logicalArraySmaller(RR, RR_or))/niter
                conta_fitness_ug = conta_fitness_ug + sum(logicalArraySE(converg, converg_or))/niter
                conta_RHO_ug = conta_RHO_ug + sum(logicalArraySE(RR, RR_or))/niter

                ck_in = 0
                for ck in range(0, niter-1):
                    if ck % step_conta == 0:
                        ck_in += 1
                        conta_mat[ck_in, 0] += converg[ck]
                        conta_mat[ck_in, 1] += converg[ck]
                        conta_mat[ck_in, 2] += RR[ck]
                        conta_mat[ck_in, 3] += RR[ck]
                if n_confro == niter/step_conta:
                    conta_mat[-1][0] += converg[-1]
                    conta_mat[-1][1] += converg[-1]
                    conta_mat[-1][2] += RR[-1]
                    conta_mat[-1][3] += RR[-1]
                
                ''' Post Processing '''
                fitness = g_x[0][-1]
                g_z_n = dente(g_x[0][0:-2], ddd-discretion*abs(ddd), uuu+discretion*abs(uuu))
                g_x_n = g_x[0][0:-1]*g_z_n
                g_x_n = g_x_n/sum(g_x_n)

                R_n = np.dot(g_x_n,np.transpose(media[0:]))
                #for t in range(0,T):
                #    prodotto_somma_n[t] = difference[t,:]+np.transpose(g_x_n)
                
                rho_n = (a/T)*(max(0,sum(sum(prodotto_somma)))) + (1-a)*(1/T**(1/b))*max(0,sum(sum(-prodotto_somma))**b)**(1/b) - R_n
                viol_somma_pesi_n = abs(sum(g_x_n)-1)
                viol_redditivita_n = max(0,np.dot(pi-g_x_n,np.transpose(media[0:])))
                viol_cardinalita_Kd_n = max(0,kd-sum(g_z_n))
                viol_cardinalita_Ku_n = max(0,sum(g_z_n)-ku)

                viol_frazione_min_n = max(0,sum(g_z_n*ddd-g_x_n))
                viol_frazione_max_n = max(0,sum(g_x_n-g_z_n*uuu))

                fitness_n = rho_n + (1/e_v[-1])*(p_v[-1,0]*viol_somma_pesi_n + p_v[-1,1]*viol_redditivita_n + p_v[-1,3]*viol_cardinalita_Kd_n + p_v[-1,4]*viol_cardinalita_Ku_n + p_v[-1,6]*viol_frazione_min_n + p_v[-1,7]*viol_frazione_max_n)
                portafoglio_n = np.transpose(g_x_n) + np.transpose(g_z_n)
                
                if giacomo == 0:
                    portfolioglobal_x_n = np.transpose(g_x_n)
                    portfolioglobal_z_n = np.transpose(g_z_n)
                    fitness_global_n = fitness_n
                    rho_globale_n = rho_n
                    rend_portaf_n = R_n
                
                ''' post processing and data saving '''
                if PSO_init == 1:
                    print('***** *****     NEW initialization with r1 = r2 = RANDOM     ***** ***** \n')
                elif PSO_init == 2:
                    print('***** *****     RANDOM initialization with r1 = r2 = 1     ***** ***** \n')
                else:
                    print('***** *****     RANDOM initialization with r1 = r2 = RANDOM     ***** ***** \n')
                
                print ('RESULTS: BEST PORTFOLIO AT THE END OF ITERATIONS (CURRENT RUN) ')
                print ('Desired yield:', pi)
                print ('Run number:', giacomo)
                print ('Number of iterations:', niter)
                print ('Fitness:', fitness)
                print ('Fitness post-processing:', abs(fitness_n[0]))
                print ('rho (risk measure):', abs(RR[-1][0]))
                print ('rho (risk measure) post-processing:',abs(rho_n))
                print ('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- ')
                viol_somma_pesi = sum(g_x[1:-2])-1
                print ('Sum weights:', sum(g_x[1:-2]))
                print ('Violation sum weights (sum_weights-1):', abs(viol_somma_pesi))
                print ('Sum post-processing weights:', sum(g_x_n))
                print ('Post-processing sum weights violation:', abs(viol_somma_pesi_n))
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- \n')
                viol_redditivita = max(0,pi-np.dot(g_x[0][:-1],np.transpose(media)))
                print('Desired income:', pi)
                print('Income obtained:', np.dot(g_x[0][:-1],np.transpose(media)))
                print('Profit violation (max (0, pi-pi_eff.)):', viol_redditivita)
                print('Income obtained post-processing:', np.dot(g_x_n,np.transpose(media[0:])))
                print('Viol. reddit. post-processing:', viol_redditivita_n)
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- \ n ')
                #not getting the intended results
                viol_cardinalita_Kd = max(0, kd-sum(g_z))
                print('Number of stocks selected:', sum(g_z))
                print('Number of selected titles post-proces.:', sum(g_z_n))
                print('Cardinality violation (kd_eff-kd_calc):', viol_cardinalita_Kd)
                print('Post-processing cardinality violation:', viol_cardinalita_Kd_n)
                viol_cardinalita_Ku = max (0, sum(g_z)-ku)
                print('Cardinality violation (ku_eff-ku_calc):', viol_cardinalita_Ku)
                print('Post-processing cardinality violation:', viol_cardinalita_Ku_n)
                viol_fraction_min = max(0, sum(g_z*ddd-g_x[0][-1]))
                viol_fraction_max = max(0, sum(g_x[0][-1]-g_z*uuu))
                print('Minimum fraction violation:', viol_fraction_min)
                print('Post-proces minimum fraction violation.:', viol_frazione_min_n)
                print('Maximum fraction violation:', viol_fraction_max)
                print('Post-proces maximum fraction violation.:', viol_fraction_max)
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- \ n ')
                print('CONTRIBUTIONS TO THE FITNESS FUNCTION (percentages) \ n')
                print('rho / Fitness:', 100 * RR[-1]/fitness)
                print('rho / Fitness post-processing:', 100 * rho_n/fitness_n)
                print('Violation sum weights / Fitness:', 100 * (abs (p_v[-1,0] * viol_somma_pesi) / e_v[-1]) / fitness)
                print('Weight sum violation / Fitness post-pr.:', 100 * (abs (p_v[-1,0] * viol_somma_pesi_n) / e_v[-1]) / fitness_n)
                print('Profitability violation / Fitness:', (100 * p_v[-1,1] * viol_redditivita / e_v[-1]) / fitness)
                print('Profitability / Fitness violation post-pr.:', (100 * p_v[-1,1] * viol_redditivita_n / e_v[-1]) / fitness_n)
                print('kd cardinality violation:', (100 * p_v[-1,3] * viol_cardinalita_Kd / e_v[-1]) / fitness)
                print('Post-proc kd cardinality violation:', (100 * p_v[-1,3] * viol_cardinalita_Kd_n / e_v[-1]) / fitness_n)
                print('Violation cardinality ku:', (100 * p_v[-1,4] * viol_cardinalita_Ku / e_v[-1]) / fitness)
                print('Post-proc ku cardinality violation:', (100 * p_v[-1,4] * viol_cardinalita_Ku_n / e_v[-1]) / fitness_n)
                print('Violation minimum fraction / Fitness:', (100 * p_v[-1,6] * viol_fraction_min / e_v[-1]) / fitness)
                print('Minimal fraction violation / Post-p Fitness:', (100 * p_v[-1,6] * viol_frazione_min_n / e_v [-1]) / fitness_n)
                print('Violation maximum fraction / Fitness:', ((100 * p_v[-1,7] * viol_fraction_max) / e_v[-1]) / fitness)
                print('Max fraction violation / Post-p Fitness:', ((100 * p_v[-1,7] * viol_frazione_max_n) / e_v[-1]) / fitness_n)
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- \ n ')

                ''' check for post-processing constraints violations '''
                if viol_redditivita_n > 0: # verifica post-processamento violazione vincolo
                    violazioni[giacomo] = 1
                
                if viol_cardinalita_Kd_n > 0: # verifica post-processamento violazione vincolo
                    violazioni[giacomo] = 2
                
                if viol_cardinalita_Ku_n > 0: # verifica post-processamento violazione vincolo
                    violazioni[giacomo] = 3
                
                if viol_frazione_min_n > 0: # verifica post-processamento violazione vincolo
                    violazioni[giacomo] = 4
                
                if viol_frazione_max_n > 0: # verifica post-processamento violazione vincolo
                    violazioni[giacomo] = 5

                if giro == 0:
                    for sat in range(0,P):
                        ARABA_z_n = dente(ARABA_ser[:,sat],ddd-discretion+abs(ddd), uuu+discretion+abs(uuu))
                        ARABA_ser[:,sat] = ARABA_ser[:,sat] * ARABA_z_n
                        ARABA_ser[:,sat] = ARABA_ser[:,sat]/sum(ARABA_ser[:,sat])
                    if giacomo == 0:
                        ARABA1 = ARABA0
                        ARABA = ARABA_ser
                
                pos_minima_or = 0
                fit_migliore_or = convergenzaglobal_or[:,pos_minima_or] # pro benchmark
                RR_migliore_or = RRglobal_or[:, pos_minima_or] # pro_benchmark
                fit_migliore = convergenzaglobal[:,pos_minima_or] # better fitness function
                RR_migliore = RRglobal[:, pos_minima_or] # better risk function
                    
                xxx = list(range(0,niter))
                fitD_su_fitS = np.divide(fit_migliore,fit_migliore_or)
                rhoD_su_rhoS = np.divide(RR_migliore,RR_migliore_or)
                vioD_su_vioS = np.divide((fit_migliore-RR_migliore),(fit_migliore_or-RR_migliore_or))
                uni = np.ones([niter,1])

                ''' comparison between portfolio before post-processing '''
                conta_fitness = conta_fitness/nrun
                conta_RHO = conta_RHO/nrun
                conta_fitness_ug = conta_fitness_ug/nrun
                conta_RHO_ug = conta_RHO_ug/nrun
                conta_mat = conta_mat/nrun

                print('\ n')
                print('\ n')
                print('\ n')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- ')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- ')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- ')
                print('COMPARISON OF DYNAMIC MANAGEMENT vs STATIC MANAGEMENT BEFORE POST-PROCESSING ')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- ')
                print('- GLOBAL COMPARISON (ON ALL ITERATIONS) ')
                print('Dynamic fitness <Static fitness:', 100 * conta_fitness)
                print('Dynamic fitness <= Static fitness:', 100 * conta_fitness_ug)
                print('dynamic RHO <static RHO:', 100 * conta_RHO)
                print('dynamic RHO <= static RHO:', 100 * conta_RHO_ug)
                dim_conta = conta_mat.shape # MIGHT HAVE THE WRONG SIZE HERE
                x_confro = np.zeros([1, dim_conta[0]]) # initialization vector for graph
                righina = 0

                for ugo in range(0,dim_conta[0]):
                    print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ')
                    if ugo == dim_conta[0]:
                        print('- COMPARISON TO THE ITERATION: ', niter)
                        x_confro[0,ugo] = niter 
                        righina = niter
                    else:
                        print('- COMPARISON TO THE ITERATION: ', ugo*step_conta)
                        x_confro[0,ugo] = ugo*step_conta 
                        righina = ugo*step_conta
                    print('Dynamic fitness < Static fitness      : ', 100*conta_mat[ugo,0])
                    print('Dynamic fitness <= Static fitness    : ', 100*conta_mat[ugo,1])

                    for fof in range(0,nrun):
                        print(convergenzaglobal[fof][0], convergenzaglobal_or[fof][0], 100*convergenzaglobal[fof][0]/convergenzaglobal_or[fof][0])
                    
                    print('Dynamic RHO <Static RHO              : ', 100*conta_mat[ugo,2])
                    print('Dynamic RHO <= Static RHO             : ', 100*conta_mat[ugo,3])
                    for fof in range(0,nrun):
                        print(RRglobal[fof][0], RRglobal_or[fof][0], 100*RRglobal[fof][0]/RRglobal_or[fof][0])
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ')
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ')

                ''' best portfolio choice among those eligible '''
                fitness_global_ser = fitness_global_n
                fitness_minimo = min(fitness_global_ser)
                rho_pre[0,giro] = RRglobal[0]
                rend_pre[0,giro] = rend_portaf
                diam_prepost[0,giro] = diametro[0][1]
                diam_prepost[1,giro] = diametro_or[0][1]

                print(' ')
                print(' ')
                print(' ')
                print('***** ***** ***** ***** **********  ***** * *****  ***** ***** ***** ***** ***** ***** ')
                print('RISULTATI: MIGLIORE PORTAFOGLIO POST-PROCESSAMENTO TRA TUTTI I RUN ')
                print('Run del portafoglio ottimo post-proces.    : ', 1)
                tab1[0,giro] = 0
                print('Fitness post-processamento                 : ', fitness_global_n[0])
                if giro == 1:
                    fitness_post_giro2 = fitness_global_n[0]
                tab1[1,giro] = fitness_global_n[0]
                print('rho (misura di rischio) post-processamento : ', rho_globale_n)
                tab1[1,giro] = rho_globale_n
                print('----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- \n')
                print('Reddito desiderato                         : ', pi)
                tab1[3,giro] = pi
                print('Reddito ottenuto post-processamento        : ', rend_portaf_n)
                tab1[4,giro] = rend_portaf_n
                print('Numero di titoli selezionati post-proces.  : ', sum(portfolioglobal_z_n[:])) #need to fix this
                tab1[5,giro] = sum(portfolioglobal_z_n[:])

                if viol_frazione_min_n > 0:
                    print('Violazione frazione minima/Fitness post-p  :           SÌ \n')
                    tab1[6,giro] = 1
                else:
                    print('Violazione frazione minima/Fitness post-p  :           NO \n')
                    tab1[6,giro] = 0
                
                if viol_frazione_max_n > 0:
                    print('Violazione frazione massima/Fitness post-p :           SÌ \n')
                    tab1[7,giro] = 1
                else:
                    print('Violazione frazione massima/Fitness post-p :           NO \n')
                    tab1[7,giro] = 0           

                print ('----- ----- ----- ----- ----- ----- ----- ----- ----- - --- ----- ----- \ n ')
                print ('Perc. run with eligible p-p portfolio: ', 100 * (1-sum(violazioni) / nrun)) #violazioni > 0
                tab1[8, giro] = 100 * (1-sum(violazioni) / nrun) #violazioni > 0
                print ('***** ***** ***** ***** ********** ***** * ***** ***** * **** ***** ***** ***** ***** ')
                print ('BEST PRE- AND POST-PROCESSING PORTFOLIO COMPOSITION BETWEEN ALL RUN (x (i) and z (i)) ')

                if (rend_portaf_n < pi) or (viol_frazione_min_n > 0) or (100*viol_frazione_max_n > 0):
                    print('W A R N I N G!!      P O R T A F O G L I O      N O N      A M M I S S I B I L E!! ')
                    am = False
                
                filename = "results" + str(today) + ".txt"

                for mostra in range(0,numvar):
                    print(abs(portfolioglobal[mostra]), abs(portfolioglobal_x_n[mostra]), portfolioglobal_z_n[mostra])
                    answer = str(abs(portfolioglobal[mostra])) + " " + str(abs(portfolioglobal_x_n[mostra])) + " " + str(portfolioglobal_z_n[mostra])
                    with open(filename, "a") as o:
                        o.write(answer)
                        o.write("\n")
                
                
                
                
                print('***** ***** ***** ***** ***** **********     The End    ***** ***** ***** ***** ***** \n')
                tab1[9,giro] = maquanti[giro] 

                for mostra in range(0,numvar):
                    tab2[mostra,(giro-1)*3+1] = portfolioglobal[mostra]
                    tab2[mostra,(giro-1)*3+2] = portfolioglobal_x_n[mostra]
                    tab2[mostra,(giro-1)*3+3] = portfolioglobal_z_n[mostra]

                #per la forntiera efficiente
                xfe[hij] = rho_globale_n
                yfe[hij] = rend_portaf_n
                if am:
                    xfe_am[hij] = rho_globale_n
                    yfe_am[hij] = rend_portaf_n


if __name__ == '__main__':
    xls = input("Enter name of xls file please:     ")
    start = time.time()
    main(xls)
    end = time.time()
    print("Total elapsed time: ", (end-start))