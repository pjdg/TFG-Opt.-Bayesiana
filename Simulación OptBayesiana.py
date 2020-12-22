
import warnings
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from skopt.benchmarks import branin


global meanz, stdz  


# Definición de configuración de simulación:
# Benchamark: 1=Branin   2=Schwefel   3=Styblinski-Tang
# Adquisición: 1=PI  2=EI  3=UCB
# Kernel: 1=RBF 2=Matern

def experimento(num_bench, num_kernel, num_adq, num_toff, name):
    
    ################# PRIMERA ETAPA: CONFIGURACIÓN DEL EXPERIMENTO #################
    
    
    benchmark = num_bench

    kernel = num_kernel

    adquisicion = num_adq

    tradeoff = num_toff

    #Función que devuelve el siguiente punto a muestrear:    

    def propose_location(acquisition, x_sample, y_sample, _gpr, _bounds, n_restarts=25):
        dim = x_sample.shape[1]
        min_val = 1e+3
        min_x = None

        #Optimización de la función objetivo para encontrar el sigueinte punto a muestrear        

        def min_obj(X):
            
            return -acquisition(X.reshape(-1, dim), x_sample, y_sample, _gpr)

    
        for x0 in np.random.uniform(_bounds[:, 0], _bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=_bounds, method='L-BFGS-B')
            if res.fun < min_val:
                # min_val = res.fun[0]
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(-1, 1)
    
    

    # DEFINICIÓN DE LA FUNCIÓN DE ADQUISICIÓN.

    if adquisicion == 1:  #PI

        def funcion_adq(X, _x_sample, _y_sample, _gpr, xi=tradeoff):
            
            mu, sigma = _gpr.predict(X, return_std=True)
            mu_sample = _gpr.predict(_x_sample)
            mu_sample_opt = np.max(mu_sample)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                pi = norm.cdf(Z)
            
            return pi


    elif adquisicion == 2:  #EI

        def funcion_adq(X, X_sample, Y_sample, gpr, xi=tradeoff):
            
            mu, sigma = gpr.predict(X, return_std=True)
            mu_sample_opt = np.max(Y_sample)

            with np.errstate(divide='warn'):
                if sigma == 0.0:
                    return 0.0
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            return ei
        

    else:  #UCB

        def funcion_adq(X, X_sample, Y_sample, gpr, xi=tradeoff):

            mu, sigma = gpr.predict(X, return_std=True)

            ucb = mu + xi * sigma

            return ucb
        
        

    # DEFINICIÓN DE LA TEST FUNCTION, FUNCIÓN DE ERROR, Y PUNTO INICIAL

    if benchmark == 1:  # Branin

        def f_original(x):

            return branin(x)

        def f_normalizada(x):
            
            return (branin(x) - meanz) / stdz 

        def f_error(x):
           
            return _z_orig - (gpr.predict(x).reshape((len(_x), len(_x))) * stdz + meanz)

        bounds = np.array([[-5, 10], [0, 15]])
        delta = (bounds[0, 1] - bounds[0, 0]) / 150

        
        _x = np.arange(bounds[0, 0], bounds[0, 1] + delta, delta)  
        _y = np.arange(bounds[1, 0], bounds[1, 1] + delta, delta)  
        _xy = np.meshgrid(_x, _y)  #
        _x = _xy[0]  
        _y = _xy[1]  
        _xy = np.array(_xy).reshape(2, -1).T  
        _z_orig = np.fromiter(map(f_original, zip(_x.flat, _y.flat)), dtype=np.float,  
                              count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)  

        meanz = np.nanmean(_z_orig)
        stdz = np.nanstd(_z_orig)
        _z = (_z_orig - meanz) / stdz

        X_init = np.array([0, 0]).T  
        Y_init = f_normalizada(X_init)

    elif benchmark == 2:  # Schwefel

        def f_original(x):

            sch2 = 418.9892 * 2 - (x[0] * np.sin(np.sqrt(abs(x[0]))) + x[1] * np.sin(np.sqrt(abs(x[1]))))

            return sch2


        def f_normalizada(x):

            sch2 = 418.9892 * 2 - (x[0] * np.sin(np.sqrt(abs(x[0]))) + x[1] * np.sin(np.sqrt(abs(x[1]))))

            return (sch2 - meanz) / stdz


        def f_error(x):
            
            return _z_orig - (gpr.predict(x).reshape((len(_x), len(_x))) * stdz + meanz)
            

        bounds = np.array([[-500, 500], [-500, 500]])
        delta = (bounds[0, 1] - bounds[0, 0]) / 150

       
        _x = np.arange(bounds[0, 0], bounds[0, 1] + delta, delta)  
        _y = np.arange(bounds[1, 0], bounds[1, 1] + delta, delta)  
        _xy = np.meshgrid(_x, _y)  
        _x = _xy[0]  
        _y = _xy[1]  
        _xy = np.array(_xy).reshape(2, -1).T  
        _z_orig = np.fromiter(map(f_original, zip(_x.flat, _y.flat)), dtype=np.float,  
                              count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)  

        meanz = np.nanmean(_z_orig)
        stdz = np.nanstd(_z_orig)
        _z = (_z_orig - meanz) / stdz

        X_init = np.array([0, -500]).T  
        Y_init = f_normalizada(X_init)

    else:  # Styblinski-Tang

        def f_original(x):

            tang2 = 0.5 * ((x[0] ** 4 - 16 * (x[0] ** 2) + 5 * x[0]) + (x[1] ** 4 - 16 * (x[1] ** 2) + 5 * x[1]))

            return tang2

        def f_normalizada(x):

            tang2 = 0.5 * ((x[0] ** 4 - 16 * (x[0] ** 2) + 5 * x[0]) + (x[1] ** 4 - 16 * (x[1] ** 2) + 5 * x[1]))

            return (tang2 - meanz) / stdz

        def f_error(x):
            
            return _z_orig - (gpr.predict(x).reshape((len(_x), len(_x))) * stdz + meanz)
            

        bounds = np.array([[-5, 5], [-5, 5]])
        delta = (bounds[0, 1] - bounds[0, 0]) / 500

        
        _x = np.arange(bounds[0, 0], bounds[0, 1] + delta, delta)  
        _y = np.arange(bounds[1, 0], bounds[1, 1] + delta, delta)  
        _xy = np.meshgrid(_x, _y)  
        _x = _xy[0] 
        _y = _xy[1]  
        _xy = np.array(_xy).reshape(2, -1).T  
        _z_orig = np.fromiter(map(f_original, zip(_x.flat, _y.flat)), dtype=np.float,  
                              count=_x.shape[0] * _x.shape[1]).reshape(_x.shape)  

        meanz = np.nanmean(_z_orig)
        stdz = np.nanstd(_z_orig)
        _z = (_z_orig - meanz) / stdz

        X_init = np.array([0, -5]).T  
        Y_init = f_normalizada(X_init)


    # DEFINICIÓN DEL KERNEL

    l_scale = 0.1 * (bounds[0, 1] - bounds[0, 0]) #Parámetro length_scale

    if kernel == 1:  # RBF

        f_kernel = ConstantKernel(1.0) * RBF(length_scale=l_scale)
        

    else:  # Màtern

        f_kernel = ConstantKernel(1.0) * Matern(length_scale=l_scale, nu=1.5) 
        
     
    #DEFINICIÓN DEL PROCESO GAUSSIANO
    
    gpr = GaussianProcessRegressor(kernel=f_kernel, alpha=1e-2)  

    
    #DEFINICIÓN DE LA FUNCIÓN DE PRECISIÓN

    def f_precision(x):
        a, b = gpr.predict(x, return_std=True)
        return b * stdz

    


    ################# SEGUNDA ETAPA: EJECUCIÓN DEL ALGORITMO #################

    n_iter = 50  # Número máximo de iteraciones (muestras)
    n_replicas = 30 #Número de réplicas del experimento

    idx_imagenes = np.arange(10, n_iter + 1, 10)  

    E = np.empty([np.size(idx_imagenes), n_replicas])  
    P = np.empty([np.size(idx_imagenes), n_replicas])

    for r in range(n_replicas):
        
        np.random.seed(r)
        
        #Las primeras muestras se corresponden con el punto inicial
        X_sample = X_init.reshape(1, -1)
        Y_sample = np.array([Y_init])

        j = 0

        fig, axs = plt.subplots(5, 2, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.07, hspace=0.2)

        for i in range(n_iter + 1):

            #Actualización del proceso gaussiano con las muestras tomadas hasta el momento
            gpr.fit(X_sample, Y_sample)

            #Obtención del siguiente punto a muestrear
            X_next = propose_location(funcion_adq, X_sample, Y_sample, gpr, bounds)
            
            #Normalización del valor obtenido en la muestra
            Y_next = f_normalizada(X_next)

            X_next = np.transpose(X_next)

            #Actualización del conjunto de datos obtenidos
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.vstack((Y_sample, Y_next))

            # Almacenamiento de datos para posterior exportación en archivo excel
            # y graficación de mapas de error y precisión
            if i in idx_imagenes:
                
                mapa_error = f_error(_xy)
                mapa_error = mapa_error.reshape((len(_x), len(_y)))

                # Error cuadrático medio
                e_cm = np.sqrt((mapa_error ** 2).sum() / (mapa_error.shape[0] * mapa_error.shape[1]))
                E[j][r] = e_cm
                

                mapa_precision = f_precision(_xy)
                mapa_precision = mapa_precision.reshape((len(_x), len(_y)))

                # Desviacion cuadrática media
                desv_cm = np.sqrt((mapa_precision ** 2).sum() / (mapa_precision.shape[0] * mapa_precision.shape[1]))
                P[j][r] = desv_cm
                
                
                # Gráficas
                graf_error = axs[j][0].pcolormesh(_x, _y, np.abs(mapa_error), shading='auto')
                axs[j][0].plot(X_sample[:, 0], X_sample[:, 1], marker="D", markersize="6", markeredgecolor="red",
                               markerfacecolor="red", linestyle='None')
                axs[j][0].axis([bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]])
                axs[j][0].annotate('n = ' + str(i), xy=(0, 0.5), xytext=(-axs[j][0].yaxis.labelpad - 5, 0),
                                   xycoords=axs[j][0].yaxis.label, textcoords='offset points',
                                   size='large', ha='right', va='center')
                axs[0][0].set_title('Error(X,Y)', fontsize=20)
                axs[j][0].set_xlabel("X")
                axs[j][0].set_ylabel("Y")
                plt.colorbar(graf_error, ax=axs[j][0])  

                graf_precision = axs[j][1].pcolormesh(_x, _y, mapa_precision, shading='auto')
                axs[j][1].plot(X_sample[:, 0], X_sample[:, 1], marker="D", markersize="6", markeredgecolor="red",
                               markerfacecolor="red", linestyle='None')
                axs[j][1].axis([bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]])
                axs[0][1].set_title('Desviación(X,Y)', fontsize=20)
                axs[j][1].set_xlabel("X")
                axs[j][1].set_ylabel("Y")
                plt.colorbar(graf_precision, ax=axs[j][1])  
                

                j += 1
            

        plt.savefig(name + '_' + str(r) + '.png')
        plt.close('all')

    excel_error = pd.DataFrame(E)
    excel_desv = pd.DataFrame(P)
    writer = pd.ExcelWriter(name + '.xlsx', engine='xlsxwriter')
    excel_error.to_excel(writer, sheet_name='Error')
    excel_desv.to_excel(writer, sheet_name='Desv')
    writer.save()

    return


f = open('log.txt', 'a')

# #E111

# f.write('Inicio E111_001' + '  ' + str(datetime.datetime.now()))
# experimento(1, 1, 1, 0.01, 'E111_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E111_01' + '  ' + str(datetime.datetime.now()))
# experimento(1, 1, 1, 0.1, 'E111_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E111_02' + '  ' + str(datetime.datetime.now()))
# experimento(1, 1, 1, 0.2, 'E111_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # ##E112

# f.write('Inicio E112_001' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 2, 0.01, 'E112_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E112_01' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 2, 0.1, 'E112_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E112_02' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 2, 0.2, 'E112_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')


# # # ##E113

# f.write('Inicio E113_001' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 3, 0.01, 'E113_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E113_01' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 3, 0.1, 'E113_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E113_02' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 1, 3, 0.2, 'E113_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')


# # # ##E121

# f.write('Inicio E121_001' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 1, 0.01, 'E121_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E121_01' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 1, 0.1, 'E121_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E121_02' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 1, 0.2, 'E121_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')


# # ##E122

# f.write('Inicio E122_001' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 2, 0.01, 'E122_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E122_01' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 2, 0.1, 'E122_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E122_02' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 2, 0.2, 'E122_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')


# # ##E123

# f.write('Inicio E123_001' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 3, 0.01, 'E123_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E123_01' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 3, 0.1, 'E123_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E123_02' + '  ' + str(datetime.datetime.now()))
# experimento( 1, 2, 3, 0.2, 'E123_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# # f.write('\n\n')


# # ##E211

# f.write('Inicio E211_001' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 1, 0.01, 'E211_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E211_01' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 1, 0.1, 'E211_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E211_02' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 1, 0.2, 'E211_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # ##E212

# f.write('Inicio E212_001' + '  ' + str(datetime.datetime.now()))
# experimento( 2, 1, 2, 0.01, 'E212_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E212_01' + '  ' + str(datetime.datetime.now()))
# experimento( 2, 1, 2, 0.1, 'E212_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E212_02' + '  ' + str(datetime.datetime.now()))
# experimento( 2, 1, 2, 0.2, 'E212_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')


# # E213

# f.write('Inicio E213_001' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 3, 0.01, 'E213_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E213_01' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 3, 0.1, 'E213_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E213_02' + '  ' + str(datetime.datetime.now()))
# experimento(2, 1, 3, 0.2, 'E213_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E221

# f.write('Inicio E221_001' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 1, 0.01, 'E221_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E221_01' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 1, 0.1, 'E221_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E221_02' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 1, 0.2, 'E221_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E222

# f.write('Inicio E222_001' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 2, 0.01, 'E222_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E222_01' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 2, 0.1, 'E222_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E222_02' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 2, 0.2, 'E222_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E223

# f.write('Inicio E223_001' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 3, 0.01, 'E223_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E223_01' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 3, 0.1, 'E223_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E223_02' + '  ' + str(datetime.datetime.now()))
# experimento(2, 2, 3, 0.2, 'E223_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E311

# f.write('Inicio E311_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 1, 0.01, 'E311_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E311_01' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 1, 0.1, 'E311_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E311_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 1, 0.2, 'E311_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E312

# f.write('Inicio E312_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 2, 0.01, 'E312_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

f.write('Inicio E312_01' + '  ' + str(datetime.datetime.now()))
experimento(3, 1, 2, 0.1, 'E312_01')
f.write('\n\n')
f.write('////////////////////////////')
f.write('\n\n')

# f.write('Inicio E312_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 2, 0.2, 'E312_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E313

# f.write('Inicio E313_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 3, 0.01, 'E313_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E313_01' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 3, 0.1, 'E313_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E313_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 1, 3, 0.2, 'E313_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E321

# f.write('Inicio E321_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 1, 0.01, 'E321_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E321_01' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 1, 0.1, 'E321_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E321_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 1, 0.2, 'E321_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E322

# f.write('Inicio E322_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 2, 0.01, 'E322_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E322_01' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 2, 0.1, 'E322_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E322_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 2, 0.2, 'E322_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# # E323

# f.write('Inicio E323_001' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 3, 0.01, 'E323_001')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E323_01' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 3, 0.1, 'E323_01')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

# f.write('Inicio E323_02' + '  ' + str(datetime.datetime.now()))
# experimento(3, 2, 3, 0.2, 'E323_02')
# f.write('\n\n')
# f.write('////////////////////////////')
# f.write('\n\n')

f.write('Finalización' + '  ' + str(datetime.datetime.now()))

f.close()