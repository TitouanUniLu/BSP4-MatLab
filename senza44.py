import pandas as pd
import datetime
from datetime import date
import numpy as np

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

    print(tab1)

    ''' Portfolio initializations'''

if __name__ == '__main__':
    main()