import pandas as pd
import datetime
from datetime import date
import numpy as np
import os
import math

current_dir = os.getcwd()

def main():

    ''' Loading the data '''
    df = pd.read_excel('TFM2.xls')
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
    nrun = 5
    maquanti = [150, 100]

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

        for giro in range (0,2):
            c1 = 1.49618
            c2 = 1.49618
            w = 0.7298
            chi = 1
            a = chi*w
            W = chi*((c1*0.5)+(c2*0.5))
            valore_max_W = 2*(a+1)

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

            if (hij == 0) and (giro == 0):
                os.mkdir(nomonef)
            
            pathmio = nomonef + '\\'
            #missing lines 152 - 158 -> don't understand why they are here bc we never created those directories (ask for help)
            fid1 = 'temp'

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

                print('******************************************************************************************* \n')
                print('******************************************************************************************* \n')
                print('*** File identifier                : \n', nomone)
                print('*** Current expected return value: \n', pi)
                print('*** round                                : \n', giro)
                if giro == 0:
                    print('*** PSO_init                            :  \n', PSO_init)            
                else:
                    print('*** PSO_init                            :  \n', 'Best population of round no. 1')
                print('*** Total number of iterations         :  \n', niter)
                print('*** Number of the current run             :  \n', giacomo)
                print('******************************************************************************************* \n')
                print('******************************************************************************************* \n')
            
                
                print(fid1, '******************************************************************************************* \n')
                print(fid1, '******************************************************************************************* \n')
                print(fid1, '*** File identifier                :  \n', nomone)

                if giro == 0:
                    print(fid1, '*** PSO_init                            :  \n', PSO_init)
                else:
                    print(fid1, '*** PSO_init                            :  \n', 'Best population of round no. 1')
                print(fid1, '*** Total number of iterations        :  \n', niter)
                print(fid1, '***Number of the current run              :  \n', giacomo)
                print(fid1, '*** Current expected return value:  \n', pi)
                print(fid1, '******************************************************************************************* \n')
                print(fid1, '******************************************************************************************* \n')

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
                e_v = np.zeros ([niter, 1]) # for dynamic epsilon storage
                v_v = np.zeros ([niter, 8]) # for dynamic constraint storage
                p_v = np.ones ([niter, 8]) # for storing dynamic weights
            
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
                vinc_1_OLD = np.zeros([P, 1]); # budget constraint
                vinc_2_OLD = np.zeros([P, 1]); # benchmark profitability constraint
                vinc_4_OLD = np.zeros([P, 1]); # benchmark cardinality constraint
                vinc_5_OLD = np.zeros([P, 1]); # benchmark cardinality constraint
                vinc_7_OLD = np.zeros([P, 1]); # constraint minimum fraction
                vinc_8_OLD = np.zeros([P, 1]); # constraint maximum fraction

                vinc_1_OLD_or = np.zeros([P, 1]); # pro benchmark
                vinc_2_OLD_or = np.zeros([P, 1]); # pro benchmark
                vinc_4_OLD_or = np.zeros([P, 1]); # pro benchmark
                vinc_5_OLD_or = np.zeros([P, 1]); # pro benchmark
                vinc_7_OLD_or = np.zeros([P, 1]); # pro benchmark
                vinc_8_OLD_or = np.zeros([P, 1]); # pro benchmark

                ''' Initialization of PSO'''
                if giro == 0:
                    if PSO_init == 1:
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
                            print('loop')

                        
            

if __name__ == '__main__':
    main()