# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import pandas as pd
from matplotlib.pyplot import figure
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import make_dir
import matplotlib as mpl

def grafico_acc_measures(path, accuracy, times):
    plt.plot(times, accuracy, color='blue', marker='o')
    plt.title('mse x tempos', fontsize=14)
    plt.xlabel('Quantidade de tempos utilizados', fontsize=14)
    plt.ylabel('Média do erro ao quadrado', fontsize=14)
    plt.grid(True)
    plt.savefig(path)
    #plt.show()
    plt.close()
    return

def gen_acc_measures(X,y, name_dpq, name):
    path = '../experimentos/%s/graficos/%s-msextempo.png'%(name_dpq, name)
    ##print("x:", np.array(X.iloc[:,:6]).shape)
    ##print("head x:", np.array(X.iloc[:,:6]))
    ##print("y:", len(np.array(y)))
    ##print("head y:", np.array(y).reshape(len(y),1))
    mse = []
    nb_measures = []

    for x_index in np.arange(start=6, stop=len(X.columns), step=6):
        mse_array = np.array([])
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(np.array(X.iloc[:,:x_index]).reshape(len(y), x_index), np.array(y).reshape(len(y),), test_size=0.3, random_state= 7)
            reg = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(X_train, y_train)
            #predict
            y_test_pred = reg.predict(X_test)
            mse_array = np.append(mse_array, mean_squared_error(y_test, y_test_pred))
        #quantidade de tempo
        qtd_tempo = int(x_index/6)

        nb_measures.append(qtd_tempo)
        mse.append(np.mean(mse_array))

    grafico_acc_measures(path, mse, nb_measures)
    #print("\nPath: %s, completo!"%path)
    return path

def plotGraph(y_t,y_p,regressorName, name, mae, mse, r2):
    make_dir("%s/graficos"%str(name))
    path = '../experimentos/%s/graficos/%s-%s.png'%(name,regressorName, name)

    try:
        y_t, y_pred=zip(*random.sample(list(zip(y_t, y_p), 150)))
       # print("Gráfico com menos de 150 pontos")
    except Exception:
        pass

    if max(y_t) >= max(y_p):
        my_range = int(max(y_t))
    else:
        my_range = int(max(y_p))
    figure(figsize=(10, 8), dpi=100)
    plt.suptitle(regressorName, y=0.95, fontsize=15)
    plt.title("%s\n"%(name)+"Média do erro absoluto: %f Média quadrada do erro: %f R2: %f" % (mae, mse, r2), fontsize=8)
    plt.scatter(range(len(y_p)), y_p, color='red', marker='*', alpha=0.8, label= "Predito")
    plt.scatter(range(len(y_t)), y_t, color='blue',marker='x', alpha=0.8, label= "Real")
    plt.ylabel('J_12')
    plt.xlabel('nº elemento')
    plt.legend(loc='upper right')
    plt.savefig(path)
    #plt.show()
    plt.close()

    plotGraph_error(y_t, y_p, regressorName, name)

    return path

#Cria um relatório para o modelo com os dados quanticos.
def quantum_report(mse, name, dpq_name, passo_shots = 1000, shots=20000):
    #Configurações de estilo do plot.
    plt.style.use('_mpl-gallery')
    mpl.rcParams['axes.titlepad'] = 2
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['text.usetex'] = True
    #mpl.rcParams['figure.figuresize'] = [4.0,4.0]
    path = '../experimentos/%s/quantum/%s-%s.png'%(dpq_name,name, dpq_name)
    #Recebe a
    x = range(passo_shots, shots+passo_shots, passo_shots)
    
    #Configurações para linha de regressão e dados.
    deg_freedom = 10
    dt = pd.DataFrame([], columns = ["x", "y"])
    dt["x"] = x
    dt["y"] = mse
    
    #Configurações para plot
    ax = sns.lmplot(x="x", y="y", data=dt,order=deg_freedom, ci=None, scatter_kws={"s": 80})
    ax.set(xlabel='Medidas Performadas', ylabel='Média do Erro ao Quadrado', title='Relação erro performance: %s\n mse:%s'%(name,str(np.mean(mse))))
        
    plt.savefig(path)
    plt.close()
    return path
    
def plotGraph_error(y_test,y_pred,regressorName, name):
    path = "../experimentos/%s/graficos/Er%s-%s.png"%(str(name),str(regressorName), str(name))
    try:
        y_test, y_pred=zip(*random.sample(list(zip(y_test, y_pred)), 5000))
    except Exception:
        pass
        #print(Exception)
        #traceback.print_exc()
        #print("Gráfico com menos de 1600 pontos")

    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))

    relative_error = []
    for y_p,y_r  in zip(y_pred, y_test):
        if y_r !=0:
            #print("Y_r:",y_r)
            relative_error.append(np.abs((y_r-y_p)/y_r))
        else:
            relative_error.append(0)

    figure(figsize=(10, 8), dpi=800)
    plt.suptitle(regressorName, y=0.95, fontsize=15)
    plt.title("%s\n"%(name), fontsize=8)
    plt.scatter(y_pred, np.array(relative_error),color='red', marker='*', alpha=0.8, label= "Predito")
    #plt.scatter(range(len(y_test)), relative_error, color='blue',marker='x', alpha=0.8, label= "Real")
    plt.ylabel('relative_error')
    plt.xlabel('real')
    plt.legend(loc='upper right')
    plt.savefig(path)
    #plt.show()
    plt.close()
    return path

def criaGraficos(dataFrame, tempos, name_dpq, name):
    make_dir("%s/graficos"%str(name_dpq))
    path = "../experimentos/%s/graficos/Obs-%s-%s.png"%(str(name_dpq),str(name), str(name_dpq))
    fig, ax = plt.subplots()
    ax.plot(tempos, dataFrame)
    ax.set(xlabel='tempo', ylabel = dataFrame.columns, title = dataFrame.columns+": "+name)
    ax.grid()
    fig.savefig(path)
    #plt.show()
    plt.close()
    return path
#Novos gráficos:
#Cria gráfico de importancia das features.
def feature_importance(features, model, name_dpq):
    path = "../experimentos/%s/graficos/FeatImp-%s.png"%(str(name_dpq), str(name_dpq))
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(12, 16), dpi=800)
    plt.title("Importância das características", fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importância Relativa',fontsize=14)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    return path

#Distribuição do erro.
def error_dist(y_real, y_pred, name, name_dpq,num_bins = 20):
    path = "../experimentos/%s/graficos/error%s-%s.png"%(str(name_dpq), str(name), str(name_dpq))
    #Erro ao quadrado
    error_sqr = (y_pred-y_real)
    #Mean of distribution
    mu = error_sqr.mean()
    #Standard deviation
    sigma = error_sqr.std()

    plt.figure(figsize=(12, 8), dpi=800)
    fig, ax = plt.subplots()
    
    #O histograma dos dados
    n, bins, patches = ax.hist(error_sqr,num_bins, density=True)
    
    #Add a 'Best fit' line
    y= ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Erro')
    ax.set_ylabel('Frequencia')
    ax.set_title(r'Distribuição do Erro: $\mu={:.2e}$, $\sigma={:.2e}$'.format(mu, sigma))
    plt.grid(True)
    #Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    #plt.show()
    plt.savefig(path)
    plt.close()
    return path

def error_Vs_j12(y_real, y_pred, name, name_dpq,intervalo=0.2):
    path = "../experimentos/%s/graficos/errorj12%s-%s.png"%(str(name_dpq), str(name), str(name_dpq))

    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    #Captura indices ordenados
    indices_ord = y_real.argsort()
    #Calculando o erro ao quadrado
    erro_sqr = (y_real[indices_ord] - y_pred[indices_ord])**2
    
    #Inicializando variáveis para o while
    error_sum = []
    error_y = list(set(y_real[indices_ord]))
    err = 0
    i_ant = indices_ord[0]
    for i in indices_ord: 
        if y_real[i_ant] == y_real[i]:
            err += erro_sqr[i] 
        else:
            error_sum.extend([err])
            err = 0
        i_ant = i
    error_sum.extend([err])
    

    plt.figure(figsize=(12, 8), dpi=800)
    plt.title(r'Erro ao Quadrado vs $J_{12}$', fontsize=20)
    plt.bar(error_y, error_sum, edgecolor="silver", linewidth=0.3, color='#AEAEB6', align='center',width=0.2)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel(r'$J_{12}$',fontsize=15)
    plt.savefig(path)
    plt.close()
    return path