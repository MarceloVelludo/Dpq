#Para construir a tabela(dataframe) que será usada para treinar o modelo vamos usar o Pandas.
import numpy as np                                     
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import random
import graphics
import pickle
import dask.array as da
from os import listdir
from os.path import isfile, join
from itertools import product
from numba import jit, vectorize 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sympy.physics.quantum import TensorProduct
from time import perf_counter 
from scipy.linalg import expm
from cmath import  *
from decimal import *
from time import perf_counter
from multiprocessing import Pool
#from math import *

#A classe DinamicaPontosQuanticos calcula a dinamica no intervalo desejado e cria a tabela com os resultados.
class DinamicaPontosQuanticos:
    """
    Classe utilizada para representar a Dinamica da equação de duplos pontos quanticos.

    ...

    Attributos
    ----------
    j_1_inicial : str
        a formatted string to print out what the animal says
    j_1_final : str
        a formatted string to print out what the animal says
    j_2_inicial : str
        a formatted string to print out what the animal says
    j_2_final : str
        a formatted string to print out what the animal says
    bz_1_inicial : str
        a formatted string to print out what the animal says
    bz_1_final : str
        a formatted string to print out what the animal says
    bz_2_inicial : str
        a formatted string to print out what the animal says
    bz_2_final : str
        a formatted string to print out what the animal says
    j_12_inicial : str
        a formatted string to print out what the animal says
    j_12_final : str
        a formatted string to print out what the animal says
    tInicial : str
        a formatted string to print out what the animal says
    tFinal : str
        a formatted string to print out what the animal says
    passoJ_1 : str
        a formatted string to print out what the animal says
    passoJ_2 : str
        a formatted string to print out what the animal says
    passoBz_1 : str
        a formatted string to print out what the animal says
    passoBz_2 : str
        a formatted string to print out what the animal says
    passoJ_12 : str
        a formatted string to print out what the animal says
    passoT : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    name_comp : str
        asdasdsa
    ro0 : numpy.array(float)[4,4]
        asdasd
    sigmaX : numpy.array(float)[2,2]
        asdasd
    sigmaY : numpy.array(float)[2,2]
        asdasd
    sigmaZ : numpy.array(float)[2,2]
        asdasd
    ident : numpy.array(float)[2,2]
        asdasd
    tensorProductIdentSigX : numpy.array(float)[4,4]
        asdasd
    tensorProductSigXIdent : numpy.array(float)[4,4]
        asdasd
    tensorProductIdentSigY : numpy.array(float)[4,4]
        asdasd
    tensorProductSigYIdent : numpy.array(float)[4,4]
        asdasd
    tensorProductIdentSigZ : numpy.array(float)[4,4]
        asdasd
    tensorProductSigZIdent : numpy.array(float)[4,4]
        asdasd
    tensorProductSigZIdentSoma : numpy.array(float)[4,4]
        asdasd
    tensorProductSigZSigZ : numpy.array(float)[4,4]
        the sound that the animal makes
    arrayJ_1 : numpy.array(float)[]
        asdasd
    arrayJ_2 : numpy.array(float)[]
        asdasd
    arrayBz_1 : numpy.array(float)[]
        asdasd
    arrayBz_2 : numpy.array(float)[]
        asdasd
    arrayJ_12 : numpy.array(float)[]
        asdasd
    arrayT : numpy.array(float)[]
        asdasd
    dataSet : pandas.DataFrame
        asdasd
    Model : sklearn.ensemble.ExtraTreesRegressor
        asdasd

    Methods
    -------
    hamiltoniana(self,j_1, j_2, bz_1, bz_2, j_12)
        Prints the animals name and what sound it makes

    hamiltoniana_T(sound=None)
        Prints the animals name and what sound it makes
    
    hamiltonianaNova(sound=None)
        Prints the animals name and what sound it makes
    
    u(sound=None)
        Prints the animals name and what sound it makes
    
    get_eigvalues(sound=None)
        Prints the animals name and what sound it makes
    
    concurrence(sound=None)
        Prints the animals name and what sound it makes
    
    ro_time_reversed(sound=None)
        Prints the animals name and what sound it makes
        
    ro(sound=None)
        Prints the animals name and what sound it makes
    
    Ox1(sound=None)
        Prints the animals name and what sound it makes
        
    Ox2(sound=None)
        Prints the animals name and what sound it makes
        
    Oy1(sound=None)
        Prints the animals name and what sound it makes
        
    Oy2(sound=None)
        Prints the animals name and what sound it makes
        
    Oz1(sound=None)
        Prints the animals name and what sound it makes
        
    Oz2(sound=None)
        Prints the animals name and what sound it makes
        
    countDecimal(sound=None)
        Prints the animals name and what sound it makes
        
    criaFrame(sound=None)
        Prints the animals name and what sound it makes
        
    calc_obs(sound=None)
        Prints the animals name and what sound it makes
        
    criaFrameNovo(self)
        Prints the animals name and what sound it makes
        
    getNames(self)
        Prints the animals name and what sound it makes
        
    criaDataFrame(self, complemento_nome = "")
        Prints the animals name and what sound it makes
        
    loadDataFrame(self, path)
        Prints the animals name and what sound it makes
        
    grafico_acc_measures(self, path, accuracy, times)
        Prints the animals name and what sound it makes
        
    gen_acc_measures(self)
        Prints the animals name and what sound it makes
        
    saveDataFrame(self, complemento_nome = "")
        Prints the animals name and what sound it makes
        
    saveDataFrameY(self, datasetY)
        Prints the animals name and what sound it makes
        
    save_Y(self, yReal, yPred, lenght)
        Prints the animals name and what sound it makes
        
    criaFrameGraficosNovo(self)
        Prints the animals name and what sound it makes
        
    criaGraficos(self, dataFrame, tempos,nome)
        Prints the animals name and what sound it makes
        
    random_float(self,low, high)(sound=None)
        Prints the animals name and what sound it makes
        
    random_samples(self,low, high, k)
        Prints the animals name and what sound it makes
        
    check_model(self)
        Prints the animals name and what sound it makes
        
    set_model(self, df = "")
        Prints the animals name and what sound it makes
        
    make_report_ox(self)
        Prints the animals name and what sound it makes
        
    make_report(self, name, y_real, y_pred)
        Prints the animals name and what sound it makes
        
    make_speed(self,t0,t1, subname = "", name = "")
        Prints the animals name and what sound it makes
        
    test_aleatorio(self, k)
        Prints the animals name and what sound it makes
        
    check_tabela(self)
        Prints the animals name and what sound it makes
        
    get_quantum_backend(self)
        Prints the animals name and what sound it makes
        
    compose_circ(j1=None, j2=None, b1=None, b2=None, j12=None, t=None)
        Prints the animals name and what sound it makes
        
        h1_circ(parametro = params['h1'])
            Prints the animals name and what sound it makes 
            
        h1_circ_2(parametro = params['h1'])
            Prints the animals name and what sound it makes
            
        h2_circ(parametro = params['h2'])
            Prints the animals name and what sound it makes 
            
        h2_circ_2(parametro = params['h2'])
            Prints the animals name and what sound it makes
            
        h3_circ1_2(parametro = params['h3'])
            Prints the animals name and what sound it makes 
            
        h3_circ2(parametro = params['h5'])
            Prints the animals name and what sound it makes
            
        h4_circ(parametro = params['h3'])
            Prints the animals name and what sound it makes 
            
        h5_circ(parametros=params['h5'])
            Prints the animals name and what sound it makes
            
        ht_circ(None)
            Prints the animals name and what sound it makes 
            
        trotSuzi_1(circ)
            Prints the animals name and what sound it makes
            
        trotSuzi_2(circ)
            Prints the animals name and what sound it makes 
            
        trotSuzi_1t(circ)
            Prints the animals name and what sound it makes
            
        trotSuzi_2t(circ)
            Prints the animals name and what sound it makes 
            
    measure_ZZ()
        Prints the animals name and what sound it makes
        
    measure_YY()
        Prints the animals name and what sound it makes 
        
    measure_YY()
        Prints the animals name and what sound it makes
        
    build_quantum_circ(params_l)
        Prints the animals name and what sound it makes
        
    exec_quantum(circs)
        Prints the animals name and what sound it makes
        
        split(a, n)
            Prints the animals name and what sound it makes 
            
    reg_qtmvscls(qt_results, cl_results)
        Prints the animals name and what sound it makes
    graph_accvsshots_maxminmedian()
        Prints the animals name and what sound it makes 
    graph_quantumvsclassic()
        Prints the animals name and what sound it makes
    make_results(self, k = 10000)
        Prints the animals name and what sound it makes 

    """
    def countDecimal2(self):
        passos = [self.passoJ_1, self.passoJ_2, self.passoBz_1,self.passoBz_2, self.passoJ_12, self.passoT]
        decimals = np.array([])
        for passo in passos:
            decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
        return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 
  
    def __init__(self, j_1_inicial=0, j_1_final=1, passoJ_1 = 0.5,
                 j_2_inicial=0, j_2_final=1, passoJ_2 = 0.5,
                 bz_1_inicial=0, bz_1_final=1, passoBz_1 = 0.5,
                 bz_2_inicial=0, bz_2_final=1, passoBz_2 = 0.5,
                 j_12_inicial=0, j_12_final=10, passoJ_12 = 0.1,
                 tInicial=1, tFinal=20, passoT=1):
        self.j_1_inicial = j_1_inicial
        self.j_1_final = j_1_final
        self.j_2_inicial = j_2_inicial
        self.j_2_final = j_2_final
        self.bz_1_inicial = bz_1_inicial
        self.bz_1_final = bz_1_final
        self.bz_2_inicial = bz_2_inicial
        self.bz_2_final = bz_2_final
        self.j_12_inicial = j_12_inicial
        self.j_12_final = j_12_final
        self.tInicial = tInicial
        self.tFinal = tFinal
        self.passoJ_1 = passoJ_1 
        self.passoJ_2 = passoJ_2 
        self.passoBz_1 = passoBz_1
        self.passoBz_2 = passoBz_2
        self.passoJ_12 = passoJ_12
        self.passoT = passoT
        
        #Para criar relatorios
        #Nome versão mais compacta e nome completo 
        self.name = str("["+str(self.j_1_inicial)[:4]+":"+str(self.j_1_final)[:4]+":"+str(self.passoJ_1)[:4]+"]"+"["+str(self.j_2_inicial)[:4]+":"+str(self.j_2_final)[:4]+":"+str(self.passoJ_2)[:4]+"]"+"["+str(self.bz_1_inicial)[:4]+":"+str(self.bz_1_final)[:4]+":"+str(self.passoBz_1)[:4]+"]"+"["+str(self.bz_2_inicial)[:4]+":"+str(self.bz_2_final)[:4]+":"+str(self.passoBz_2)[:4]+"]"+"["+str(self.j_12_inicial)[:4]+":"+str(self.j_12_final)[:4]+":"+str(self.passoJ_12)[:4]+"]"+"["+str(self.tInicial)[:3]+":"+str(self.tFinal)[:3]+":"+str(self.passoT)[:4]+"]")
        
        self.name_comp = str("["+str(self.j_1_inicial)+":"+str(self.j_1_final)+":"+str(self.passoJ_1)+"]"+"["+str(self.j_2_inicial)+":"+str(self.j_2_final)+":"+str(self.passoJ_2)+"]"+"["+str(self.bz_1_inicial)+":"+str(self.bz_1_final)+":"+str(self.passoBz_1)+"]"+"["+str(self.bz_2_inicial)+":"+str(self.bz_2_final)+":"+str(self.passoBz_2)+"]"+"["+str(self.j_12_inicial)+":"+str(self.j_12_final)+":"+str(self.passoJ_12)+"]"+"["+str(self.tInicial)+":"+str(self.tFinal)+":"+str(self.passoT)+"]")
        
        #roInicial
        #UpUp
        #self.ro0 = np.array([[1,0,0,0], [0,0,0,0], [0,0,0,0],[0,0,0,0]])
        self.ro0 = np.array([[0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25], [0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])
        #Criando as matrizes de Pauli-X, Pauli-Y, e Pauli-Z.
        self.sigmaX = np.array([[0, 1], [1, 0]])
        self.sigmaY = np.array([[0, -1j], [1j, 0]])
        self.sigmaZ = np.array([[1, 0], [0, -1]])
        
        #Matriz identidade.
        self.ident = np.identity(2)
        
        #Algumas constantes que são usadas diversas vezes para os calculos.
        self.tensorProductIdentSigX = TensorProduct(self.ident, self.sigmaX)
        self.tensorProductSigXIdent = TensorProduct(self.sigmaX, self.ident)
        self.tensorProductIdentSigY = TensorProduct(self.ident, self.sigmaY)
        self.tensorProductSigYIdent = TensorProduct(self.sigmaY, self.ident)
        self.tensorProductIdentSigZ = TensorProduct(self.ident, self.sigmaZ)
        self.tensorProductSigZIdent = TensorProduct(self.sigmaZ, self.ident)
        self.tensorProductSigZIdentSoma = TensorProduct(self.sigmaZ, self.ident) + TensorProduct(self.ident, self.sigmaZ)
        self.tensorProductSigZSigZ = TensorProduct(self.sigmaZ, self.sigmaZ)
        
        decimalJ_1, decimalJ_2, decimalBz_1, decimalBz_2, decimalJ_12, decimalT = self.countDecimal2()
        
        self.arrayJ_1 = (np.arange(self.j_1_inicial*decimalJ_1,decimalJ_1*self.j_1_final+self.passoJ_1*decimalJ_1, self.passoJ_1*decimalJ_1)/decimalJ_1).tolist()
        self.arrayJ_2 = (np.arange(self.j_2_inicial*decimalJ_2, decimalJ_2*self.j_2_final+self.passoJ_2*decimalJ_2, self.passoJ_2*decimalJ_2)/decimalJ_2).tolist()
        self.arrayBz_1 = (np.arange(self.bz_1_inicial*decimalBz_1, decimalBz_1*self.bz_1_final+self.passoBz_1*decimalBz_1, self.passoBz_1*decimalBz_1)/decimalBz_1).tolist()
        self.arrayBz_2 = (np.arange(self.bz_2_inicial*decimalBz_2, decimalBz_2*self.bz_2_final+self.passoBz_2*decimalBz_2, self.passoBz_2*decimalBz_2)/decimalBz_2).tolist()
        self.arrayJ_12 = (np.arange(self.j_12_inicial*decimalJ_12, decimalJ_12*self.j_12_final+self.passoJ_12*decimalJ_12, self.passoJ_12*decimalJ_12)/decimalJ_12).tolist()
        self.arrayT = (np.arange(self.tInicial*decimalT, decimalT*self.tFinal+self.passoT*decimalT, self.passoT*decimalT)/decimalT).tolist()
        self.elementos_iter = list(product(self.arrayJ_1, self.arrayJ_2, self.arrayBz_1, self.arrayBz_2,  self.arrayJ_12))

        self.dataSet = None
        self.model = None
        
     #Definição da equação da dinâmica de pontos quanticos versão mais completa.
    def hamiltoniana(self,j_1, j_2, bz_1, bz_2, j_12):
        ##print("parametros hamiltoniana:", j_1, j_2, bz_1, bz_2, j_12)
        #input()
        return  0.5*(np.multiply(j_1, self.tensorProductSigZIdent) + np.multiply(j_2,self.tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(self.tensorProductSigZSigZ - self.tensorProductSigZIdent - self.tensorProductIdentSigZ)) + np.multiply(bz_1,self.tensorProductSigXIdent) + np.multiply(bz_2,self.tensorProductIdentSigX))
    
#------------------------------------------------------------------------------------------------
#Divisão do hamiltoniano
    def hamiltoniana_p(self,j_1, j_2, bz_1, bz_2, j_12):
        return (0.5*np.multiply(j_1, self.tensorProductSigZIdent), 0.5*np.multiply(j_2,self.tensorProductIdentSigZ), 0.5*0.5*np.multiply(j_12,(self.tensorProductSigZSigZ - self.tensorProductSigZIdent - self.tensorProductIdentSigZ)) , 0.5*np.multiply(bz_1,self.tensorProductSigXIdent) , 0.5*np.multiply(bz_2,self.tensorProductIdentSigX))
    
    def hamiltoniana_T(self,j_12):
        return 0.5*0.5*np.multiply(j_12,-self.tensorProductSigZIdent)   
#-------------------------------------------------------------------------------------------------------    
    #Definição da equação da dinâmica de pontos quanticos versão mais completa.
    #@vectorize(target="cuda")
    def hamiltonianaNova(self, parametros):
        j_1 = parametros[0]
        j_2 = parametros[1]
        bz_1 = parametros[2]
        bz_2 = parametros[3] 
        j_12 = parametros[4]
        ##print("parametros hamiltoniana:",parametros)
        #input()
        return  0.5*(np.multiply(j_1, self.tensorProductSigZIdent) + np.multiply(j_2,self.tensorProductIdentSigZ) + 0.5*np.multiply(j_12,(self.tensorProductSigZSigZ - self.tensorProductSigZIdent - self.tensorProductIdentSigZ)) + np.multiply(bz_1,self.tensorProductSigXIdent) + np.multiply(bz_2,self.tensorProductIdentSigX))

    
    #Definindo a função operador temporal.
    def u(self,t, h):
        h = h.astype(np.clongdouble)
        t = np.array([t]).astype(np.clongdouble)
        eq1 = np.multiply(h,t)
        eq2 = np.multiply(eq1,(-1j))
        #eq3 = np.multiply(-1, eq2)
        #result = expm((np.matmul(np.matmul(h,t),(-1j))))
        #result = expm(eq3)
        ##print('result:', result)
        return expm(np.multiply((np.multiply(h,t)),(-1j)))
    
    #Retorna o eigenvalues da multiplicação de ro com ro tempo reverso
    def get_eigvalues(self, ro, ro_tr):
        eigvalues, eigvectors = np.linalg.eig(np.matmul(ro,ro_tr))
        return eigvalues
    
    #Retorna a medida da concorrencia dado o ro.
    def concurrence(self, ro):
        ro_tr = self.ro_time_reversed(ro)
        eig_val = self.get_eigvalues(ro, ro_tr)
        eig_sqr_ord = np.sqrt(np.sort(eig_val)[::-1])
        eig_sum = eig_sqr_ord[0]
        for eig_sqrt in eig_sqr_ord[1:]:
            eig_sum -= eig_sqrt
        return np.maximum(0, eig_sum)
    
    #Definindo a função que calcula o Operador Densidade tempo-reverso 
    def ro_time_reversed(self, ro):
        tp_sigy_sigy = TensorProduct(self.sigmaY, self.sigmaY)
        ro_conj = np.conjugate(ro)
        return np.matmul(tp_sigy_sigy , np.matmul(ro_conj, tp_sigy_sigy))
    
    #Definindo a função operador densidade.
    def ro(self,t, h):
        #print("t:",t)
        #print("\ntype:", type(t))
        u = self.u(t, h)
        ##print("u:\n", u)
        ##print("matrix:\n", np.matrix(u))
        ##print("matrix dagger:\n", np.matrix(u).getH())
        #input()
        return np.dot(np.dot(u,self.ro0), np.array(np.matrix(u).getH()))


    #--------------------------------------------------
    #Observaveis:
    #Definindo a função O^(1)_x 
    def Ox1(self,ro):
        a = np.dot(self.tensorProductSigXIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_x 
    def Ox2(self,ro):
        a = np.dot(self.tensorProductIdentSigX, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_y 
    def Oy1(self,ro):
        a = np.dot(self.tensorProductSigYIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_y 
    def Oy2(self,ro):
        a = np.dot(self.tensorProductIdentSigY, ro)
        return np.trace(a)

    #--------------------------------------------------
    #Definindo a função O^(1)_z 
    def Oz1(self,ro):
        a = np.dot(self.tensorProductSigZIdent, ro)
        return np.trace(a)


    #Definindo a função O^(2)_z 
    def Oz2(self,ro):
        a = np.dot(self.tensorProductIdentSigZ, ro)
        return np.trace(a)
        
    def countDecimal(self):
        passos = [self.passoJ_1, self.passoJ_2, self.passoBz_1,self.passoBz_2, self.passoJ_12, self.passoT]
        decimals = np.array([])
        for passo in passos:
            decimals = np.append(decimals, 10**(-Decimal(str(passo)).as_tuple().exponent))
        return (decimals[0], decimals[1], decimals[2], decimals[3], decimals[4], decimals[5]) 
    

    #@vectorize(target="cuda")
    def criaFrame(self):
        #t0 = perf_counter()
        results = np.array([])
        t1 = t0 = perf_counter()
        j_12_len = len(self.arrayJ_12)
        for index,j12Dez in enumerate(self.arrayJ_12):
            ##print("{:.1f}\n".format(index/j_12_len))
            ##print("Total tempo gasto: ", t1 - t0)
            #t0 = perf_counter()
            j_12 = j12Dez
            for j1Dez in self.arrayJ_1:
                j_1 = j1Dez
                for j2Dez in self.arrayJ_2:
                    j_2 = j2Dez
                    for bz1Dez in self.arrayBz_1:
                        bz_1 = bz1Dez 
                        for bz2Dez in self.arrayBz_2:
                            bz_2 = bz2Dez
                            resultsOx = np.array([])
                            hvalor = self.hamiltoniana(j_1, j_2, bz_1, bz_2, j_12)
                            for tDez in self.arrayT:
                                t = tDez
                                rovalor = self.ro(t,hvalor)
                                ox1 = np.float32(self.Ox1(rovalor))
                                ox2 = np.float32(self.Ox2(rovalor))
                                oy1 = np.float32(self.Oy1(rovalor))
                                oy2 = np.float32(self.Oy2(rovalor))
                                oz1 = np.float32(self.Oz1(rovalor))
                                oz2 = np.float32(self.Oz2(rovalor))
                                resultsOx =  np.append(resultsOx,[ox1, ox2, oy1, oy2, oz1, oz2])
                            resultsOx = np.append([j_1, j_2, bz_1, bz_2, tDez], resultsOx)
                            results = np.append(results, resultsOx)
        t1 = perf_counter()
        #t1 = perf_counter()
        colunas = int((((((self.tFinal - self.tInicial)/self.passoT)+1)*6)+5))
        linhas = int(len(results)/colunas)
        
        #print('colunas:', colunas)
        #print("Total tempo gasto: ", t1 - t0)   
        #print("results shape:", results.shape)
        #print("Tamanho:", len(results))
        #print('linhas:', linhas)
        return np.float32(results.reshape(linhas, colunas))
    
    #@vectorize(target="cuda")
    def calc_obs(self, hvalor):
        resultsOx = []
        for t in self.arrayT:
            rovalor = self.ro(t,hvalor)
            ox1 = np.float32(self.Ox1(rovalor))
            ox2 = np.float32(self.Ox2(rovalor))
            oy1 = np.float32(self.Oy1(rovalor))
            oy2 = np.float32(self.Oy2(rovalor))
            oz1 = np.float32(self.Oz1(rovalor))
            oz2 = np.float32(self.Oz2(rovalor))
            resultsOx.append([ox1, ox2, oy1, oy2, oz1, oz2])
        return resultsOx
    
    #@vectorize(target="cuda")
    def criaFrameNovo(self):
        ##print(len(self.elementos_iter))
        t1 = t0 = perf_counter()
        
        reslts_hvalor =list(map(self.hamiltonianaNova, self.elementos_iter))
        #print("head resultados:", reslts_hvalor[:3])
        resultsOxJ = list(map(self.calc_obs, list(reslts_hvalor)))
        
        t1 = perf_counter()
        colunas = int(((((self.tFinal - self.tInicial)/self.passoT)+1)*6))
        linhas = int(len(self.elementos_iter))
        #print("elementos_iter_array: ", self.elementos_iter[:5])
        ##print('colunas:', colunas)
        #print("Total tempo gasto: ", t1 - t0)   
        ##print("results shape:", results.shape)
        #print("Tamanho:", len(resultsOxJ))
        #print("resultados:", resultsOxJ[:5])
        ##print('linhas:', linhas)
        
        #Compila o resultado com os elementos referentes a cada resultado
        result = np.hstack((np.reshape(np.array(self.elementos_iter), (linhas, 5)),np.reshape(np.array(resultsOxJ), (linhas, colunas))))
        #print(result.shape)
        return result
    
    #dataframe
    def getNames(self):
        listO = [['ox1T' + str(tempo),'ox2T' + str(tempo), 'oy1T' + str(tempo), 'oy2T' + str(tempo), 'oz1T' + str(tempo),'oz2T' + str(tempo)] for tempo in self.arrayT]
        listOFlat = np.array([])
        for tempos in listO:
            listOFlat = np.append(listOFlat, np.array(tempos))
        return np.append(np.append(np.append(np.append(['j_1'],['j_2']),np.append(['bz_1'], ['bz_2'])), ['j_12_Target']) , listOFlat)
    
    def criaDataFrame(self, complemento_nome = ""):
        self.dataSet = pd.DataFrame(self.criaFrameNovo(), columns = self.getNames())
        self.saveDataFrame(complemento_nome)
        return self.dataSet
    
    def loadDataFrame(self, path):
        self.dataSet = pd.read_csv(path, index_col = 0)
        return self.dataSet
    
    def grafico_acc_measures(self, path, accuracy, times):
        plt.plot(times, accuracy, color='blue', marker='o')
        plt.title('mse x tempos', fontsize=14)
        plt.xlabel('Quantidade de tempo utilizado', fontsize=14)
        plt.ylabel('média do erro ao quadrado', fontsize=14)
        plt.grid(True)
        plt.savefig("results/"+self.name+"-msextempo"+".png")
        plt.close()
        return
    
    def gen_acc_measures(self):
        path = "data/TabelasNovas/"
        X = self.dataSet.loc[:,self.dataSet.columns.str.startswith('o')]
        y = self.dataSet.loc[:,self.dataSet.columns.str.startswith('j_12')]
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
        
        self.grafico_acc_measures(path, mse, nb_measures)
        #print("\nPath: %s, completo!"%path)
        return
        
    def saveDataFrame(self, complemento_nome = ""):
        if self.dataSet is None:
             self.criaDataFrame()
        self.dataSet.to_csv(path_or_buf="./data/TabelasNovas/"+complemento_nome+ self.name + ".csv")
        return
    
    def saveDataFrameY(self, datasetY):
        datasetY.to_csv(path_or_buf="./data/TabelasNovas/Y" +"yRealyPred -"+ self.name + ".csv")
        return
    
    def save_Y(self, yReal, yPred, lenght):
        #print("shape:", np.array(yReal).reshape((lenght,-1)).shape)
        yReal = np.array(yReal).reshape((lenght,-1))
        yPred = np.array(yPred).reshape((lenght,-1))
        datasetY = pd.DataFrame(np.hstack([yReal, yPred]), columns = ["yReal","yPred"])
        self.saveDataFrameY(datasetY)
        return datasetY
    #Cria frame para observaveis.
    def criaFrameGraficosNovo(self):
        t0 = perf_counter()
        results = np.array([])
        ox1 = np.array([])
        ox2 = np.array([])
        oy1 = np.array([])
        oy2 = np.array([])
        oz1 = np.array([])
        oz2 = np.array([])
        tempos = np.array([])
        for j_12 in self.arrayJ_12:
            for j_1 in self.arrayJ_1:
                for j_2 in self.arrayJ_2:
                    for bz_1 in self.arrayBz_1:
                        for bz_2 in self.arrayBz_2:
                            resultsOx = np.array([])
                            hvalor = self.hamiltoniana(j_1, j_2, bz_1, bz_2, j_12)
                            for t in self.arrayT:
                                rovalor = self.ro(t,hvalor)
                                ox1 = np.float32(np.append(ox1, self.Ox1(rovalor)))
                                ox2 = np.float32(np.append(ox2, self.Ox2(rovalor)))
                                oy1 = np.float32(np.append(oy1, self.Oy1(rovalor)))
                                oy2 = np.float32(np.append(oy2, self.Oy2(rovalor)))
                                oz1 = np.float32(np.append(oz1, self.Oz1(rovalor)))
                                oz2 = np.float32(np.append(oz2, self.Oz2(rovalor)))
                                tempos = np.append(tempos, t)

        t1 = perf_counter()

        return pd.DataFrame(ox1, columns = ['ox1']), pd.DataFrame(ox2, columns = ['ox2']), pd.DataFrame(oy1, columns = ['oy1']), pd.DataFrame(oy2, columns = ['oy2']), pd.DataFrame(oz1, columns = ['oz1']), pd.DataFrame(oz2, columns = ['oz2']), pd.DataFrame(tempos, columns = ['tempo'])

    def criaGraficos(self, dataFrame, tempos,nome):
        fig, ax = plt.subplots()
        ax.plot(tempos, dataFrame)
        ax.set(xlabel='tempo', ylabel = dataFrame.columns, title = dataFrame.columns)
        ax.grid()
        fig.savefig(nome+".png")
        #plt.show()
        return

   
    
    ### modelo
    def check_model(self):
        #Path to tables directory.
        path = "data/Models/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        matching = [s for s in onlyfiles if self.name in s]
        #print("matching:\n", matching)
        if not matching:
            return True
        else:
            self.loadDataFrame(matching[0])
            return False
        
    def set_model(self, df = ""):
        if df == "":
            df = self.dataSet
        #print("df:\n", df)
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,5:], df.iloc[:,4], test_size=0.3, random_state= 7)
        #Se o modelo ja existir carrega ele, se não faz um novo e guarda ele.
        if(self.check_model):
            self.model = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(X_train, y_train)
            pickle.dump(self.model, open("./data/Models/"+self.name, 'wb'))
        else:
            self.model = pickle.load(open("./data/Models/"+self.name, 'rb'))
            
        return self.model, X_train, X_test, y_train, y_test
    
    #Gera relatorio das observaveis em relação ao tempo.
    def make_report_ox(self):
        #Para os pontos minimos
        dpqMin = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_inicial, j_1_final=self.j_1_inicial, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_inicial, j_2_final=self.j_2_inicial, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_inicial, bz_1_final=self.bz_1_inicial, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_inicial, bz_2_final=self.bz_2_inicial, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_inicial, j_12_final=self.j_12_inicial, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMin.criaFrameGraficos()
        dpqMin.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMin.name)
        dpqMin.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMin.name)
        dpqMin.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMin.name)
        dpqMin.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMin.name)
        dpqMin.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMin.name)
        dpqMin.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMin.name)
        
        #Para os pontos Maximos
        dpqMax = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_final, j_1_final=self.j_1_final, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_final, j_2_final=self.j_2_final, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_final, bz_1_final=self.bz_1_final, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_final, bz_2_final=self.bz_2_final, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_final, j_12_final=self.j_12_final, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMax.criaFrameGraficos()
        dpqMax.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMin.name)
        dpqMax.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMin.name)
        dpqMax.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMin.name)
        dpqMax.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMin.name)
        dpqMax.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMin.name)
        dpqMax.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMin.name)
        
        #Para os pontos médios
        dpqMean = DPQ.DinamicaPontosQuanticos(j_1_inicial=self.j_1_final/2, j_1_final=self.j_1_final/2, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_final/2, j_2_final=self.j_2_final/2, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_final/2, bz_1_final=self.bz_1_final/2, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_final/2, bz_2_final=self.bz_2_final/2, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_final/2, j_12_final=self.j_12_final/2, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMean.criaFrameGraficos()
        dpqMean.criaGraficos(ox1, tempos, "Observável ox1- " + dpqMin.name)
        dpqMean.criaGraficos(ox2, tempos, "Observável ox2- " + dpqMin.name)
        dpqMean.criaGraficos(oy1, tempos, "Observável oy1- " + dpqMin.name)
        dpqMean.criaGraficos(oy2, tempos, "Observável oy2- " + dpqMin.name)
        dpqMean.criaGraficos(oz1, tempos, "Observável oz1- " + dpqMin.name)
        dpqMean.criaGraficos(oz2, tempos, "Observável oz2- " + dpqMin.name)
        
        return

    def make_report(self, name, y_real, y_pred):
        relative_error_1 = 0
        relative_error_2 = 0

        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real,y_pred)

        for y_p, y_t in zip(y_pred, y_real):
            relative_error_1 += np.abs(1-y_p)/y_t
            relative_error_2 += np.abs((y_t-y_p)/y_t)

        with open("Resultados.txt", "a+") as text_file:
            text_file.write("===="*5+"\n")
            text_file.write(name)
            text_file.write("\n%s"%(self.name_comp))
            text_file.write("\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f\nrelative_error_1: %f \nrelative_error_2: %f\n" % (mae, mse, r2, relative_error_1, relative_error_2))

        graphics.plotGraph(y_real, y_pred, "Extra Trees Regressor "+name,self.name, mae, mse, r2)
        self.gen_acc_measures()
        
        return

    def make_speed(self,t0,t1, subname = "", name = ""):
        if name == "":
            name = self.name_comp
        with open("Speed.txt", "a+") as text_file:
            text_file.write("===="*5)
            text_file.write(subname+"\n")
            text_file.write("\n%s"%(name))
            text_file.write("Tempo: %f \n" % (t1-t0))
        return

    def test_aleatorio(self, k):
        arrayJ_1 = self.random_samples(self.j_1_inicial, self.j_2_final, 10)
        arrayJ_2 = self.random_samples(self.j_2_inicial, self.j_2_final, 10)
        arrayBz_1 = self.random_samples(self.bz_1_inicial, self.bz_1_final, 10)
        arrayBz_2 = self.random_samples(self.bz_2_inicial, self.bz_2_final, 10)
        arrayJ_12 = self.random_samples(self.j_12_inicial, self.j_12_final, 10)

        self.elementos_iter = random.sample(list(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12)), k)
        #print("\n len teste aleatorio:", len(self.elementos_iter))
        df = self.criaDataFrame("teste-k-")
        X_val = df.iloc[:,5:]
        y_val = df.iloc[:,4]
        y_val_pred = self.model.predict(X_val)
        #print("len y_val_pred:",len(y_val_pred))

        self.make_report("teste aleatorio com K: %f "%k, y_val, y_val_pred)

        return
    
    #Check if the table requested is in the base
    def check_tabela(self):
        #Path to tables directory.
        path = "data/TabelasNovas/"
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        matching = [s for s in onlyfiles if (self.name in s) and (not 'y' in s) and (not "k" in s) ]
        #print("files:", onlyfiles)
        #print("\nMatching:", matching)
        if not matching:
            return True
        else:
            self.loadDataFrame(path+matching[0])
            return False
    ##-----------------
    ## Parte Quantica!!
    ##-----------------
    from qiskit.providers.ibmq import least_busy

    #Seleciona dentre os backends pequenos o que estiver mais desocupado.
    def get_quantum_backend():    
        small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                          and not x.configuration().simulator)
        backend = least_busy(small_devices)
        return backend

    #Monta circuito de acordo com hamiltoniana.
    def compose_circ(j1=None, j2=None, b1=None, b2=None, j12=None, t=None):
        #Trotter-suzuki parameters:
        # n-Numero de divisões
        n = 1
        # delt_t- divisão do tempo por 
        delt_j1 = (j1*t)/n
        delt_j2 = (j2*t)/n
        delt_b1 = (b1*t)/n
        delt_b2 = (b2*t)/n
        delt_j12 = (j12*t)/(2*n)

        #pi
        pi =3.1415

        #Inicializa circuito.
        circ_h = QuantumCircuit(2,2)

        #Inicializa os dois qubits no estado inicial.
        circ_h.h(0)
        circ_h.h(1)

        #Parametros para os gates
        params_ = {'h1': delt_j1, 'h2': delt_j2, 'h3': delt_j12, 'h4': delt_b1, 'h5': delt_b2}
        params = {'h1': Parameter('h1'), 'h2': Parameter('h2'), 'h3': Parameter('h3'), 'h4': Parameter('h4'), 'h5': Parameter('h5')}


        #Parte h1
        def h1_circ(parametro = params['h1']):
            circ_h1 = QuantumCircuit(2,2)
            circ_h1.barrier([0,1])
            #circ_h1.h(1)
            circ_h1.cnot(1,0)
            circ_h1.rz(parametro,1)
            circ_h1.cnot(1,0)
            #circ_h1.h(1)
            #circ_h1.i(1)
            circ_h1.barrier([0,1])
            return circ_h1
        #Parte h1
        def h1_circ_2(parametro = params['h1']):
            circ_h1 = QuantumCircuit(2,2)
            circ_h1.barrier([0,1])
            #circ_h1.h(1)
            circ_h1.rz(parametro,1)
            #circ_h1.h(1)
            #circ_h1.i(1)
            circ_h1.barrier([0,1])
            return circ_h1


        #Parte h2
        def h2_circ(parametro = params['h2']):
            circ_h2 = QuantumCircuit(2,2)
            circ_h2.barrier([0,1])
            circ_h2.cnot(0,1)
            circ_h2.rz(parametro,0)
            circ_h2.cnot(0,1)
            circ_h2.barrier([0,1])
            return circ_h2

        #Parte h2
        def h2_circ_2(parametro = params['h2']):
            circ_h2 = QuantumCircuit(2,2)
            circ_h2.barrier([0,1])
            circ_h2.rz(parametro,0)
            circ_h2.barrier([0,1])
            return circ_h2

        #Parte h3-1
        def h3_circ1_2(parametro = params['h3']):        
            circ_h3 = QuantumCircuit(2,2)
            circ_h3.barrier([0,1])
            circ_h3.rzz(parametro, 1, 0)
            circ_h3.barrier([0,1])
            return circ_h3

        #Parte h3-2
        def h3_circ2(parametro = params['h3']):
            return h1_circ_2(-parametro)
        #Parte h3-3
        def h3_circ3(parametro = params['h3']):        
            return h2_circ_2(-parametro)


        #Parte h4
        def h4_circ(parametros=params['h4']):
            circ_h4 = QuantumCircuit(2,2)
            circ_h4.barrier([0,1])
            circ_h4.h(1)
            circ_h4.compose(h1_circ(parametros),inplace=True)#rz? estimar theta com j12?
            circ_h4.h(1)
            circ_h4.barrier([0,1])
            return circ_h4

        #parte h5
        def h5_circ(parametros=params['h5']):
            circ_h5 = QuantumCircuit(2,2)
            circ_h5.barrier([0,1])
            circ_h5.h(0)
            circ_h5.compose(h2_circ(parametros),inplace=True)#rz? estimar theta com j12?
            circ_h5.h(0)
            circ_h5.barrier([0,1])
            return circ_h5

        #Constroi circuito para Hamiltoniana completa
        def ht_circ():
            circ_t = QuantumCircuit(2,2)
            circ_t.compose(h1_circ_2(), inplace=True)
            circ_t.compose(h2_circ_2(), inplace=True)
            circ_t.compose(h3_circ1_2(), inplace=True)
            circ_t.compose(h3_circ2(), inplace=True)
            circ_t.compose(h3_circ3(), inplace=True)
            #circ_t.compose(h4_circ(), inplace=True)
            #circ_t.compose(h5_circ(), inplace=True)
            return circ_t


        #Trotter-suzuki 1 ordem.
        def trotSuzi_1(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            #_ = list(map(ts.compose(circ, inplace=True),range(n)))
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return

        #Trotter-suzuki 2 ordem.
        def trotSuzi_2(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            ts.compose(circ, inplace=True)
            ts_rev = ts.reverse_ops()
            ts.compose(ts_rev, inplace=True)
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return 

        #Trotter-suzuki 1 ordem para teste.
        def trotSuzi_1t(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            #_ = list(map(ts.compose(circ, inplace=True),range(n)))
            for a in range(n):
                ts.compose(circ, inplace=True)
            circ_h.compose(ts, inplace=True)
            return

        #Trotter-suzuki 2 ordem para teste.
        def trotSuzi_2t(circ):
            #Cria circuito da trotter suzuki de primeira ordem.
            ts = QuantumCircuit(2,2)
            for a in range(n):
                ts.compose(circ, inplace=True)
            ts_rev = ts.reverse_ops()
            circ_h.compose(ts, inplace=True)
            circ_h.compose(ts_rev, inplace=True)
            return 

        circ_t = ht_circ()
        circ_h.compose(circ_t,inplace=True)

        #trotSuzi_1(circ_t.assign_parameters([delt_j1,delt_j2,delt_j12,delt_b1,delt_b2]))
        #trotSuzi_2(circ_t.assign_parameters([delt_j1/2,delt_j2/2,delt_j12/2,delt_b1/2,delt_b2/2]))
        circ_h.assign_parameters([delt_j1,delt_j2,delt_j12], inplace=True)
        #circ_h.measure_all(add_bits=False)
        #circ_h.save_expectation_value(Z, [0])
        return circ_h

    #Constroi circuito para medida ZZ
    def measure_ZZ():
        circ_zz = QuantumCircuit(2,2)
        circ_zz.measure_all(add_bits=False)
        return circ_zz

    #Constroi circuito para medida YY
    def measure_YY():
        circ_yy = QuantumCircuit(2,2)
        circ_yy.rx(pi/2,0) #transforma |y+> em |0> e |y-> em |1> para efetuar medida em X
        circ_yy.rx(pi/2,1)
        circ_yy.measure_all(add_bits=False)
        return circ_yy

    #Constroi circuito para medida XX
    def measure_XX():
        circ_xx = QuantumCircuit(2,2)
        circ_xx.h(0)
        circ_xx.h(1)
        circ_xx.measure_all(add_bits=False)
        return circ_xx

    #Constroi os circuitos para execução
    def build_quantum_circ(params_l):
        circs_l = []
        for params in params_l:
            for t in self.arrayT:
                circ_base = compose_circ(j1=params[0], j2=params[1], b1=params[2], b2=params[3], j12=params[4], t=t)
                circ_xx = circ_base.compose(measure_XX())
                circ_yy = circ_base.compose(measure_YY())
                circ_zz = circ_base.compose(measure_ZZ())
                circs_l.append([circ_xx, circ_yy, circ_zz])            
        return circs_l

    #Coloca em execução no computador quantico a lista de circuitos e aguarda pelo seu resultado.
    def exec_quantum(circs):
        #Captura a instancia quantica que está mais desocupada
        backend = self.get_Quantum_backend()
        circs = transpile(circs, backend=backend)
        # Use Job Manager to break the circuits into multiple jobs.
        job_manager = IBMQJobManager()
        job_set_DPQ = job_manager.run(circs, backend=backend, name='DPQ', shots = self.shots)

        #Espera até que o circuito seja executado
        while(job_set_DPQ.status() is JobStatus.RUNNING):
            try:
                job_status = job_set_DPQ.status()  # Query the backend server for job status.
                if job_status is JobStatus.RUNNING:
                    print("The job is still running")
            except IBMQJobApiError as ex:
                print("Something wrong happened!: {}".format(ex))

        #Captura os resultados do circuito
        try:
            job_result = job_set_dpq.result()  # It will block until the job finishes.
            print("The job finished with result {}".format(job_result))
        except JobError as ex:
            print("Something wrong happened!: {}".format(ex))

        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

        return split(job_result, 3)

    #Prediz os resultados para o classico e quantico.
    def reg_qtmvscls(qt_results, cl_results):
        #Seta variaveis
        y_real = cl_results.loc[:,cl_results.columns == 'j_12_Target']
        X_cls = cl_results.loc[:,cl_results.columns != 'j_12_Target']
        X_qt = qt_results

        #Faz as predições
        y_cl_pred = self.model.predict(cl_results['j_12_Target'])
        y_qt_pred = self.model.predict(qt_results)

        return  y_cl_pred, y_qt_pred

    def graph_accvsshots_maxminmedian():
        return

    def graph_quantumvsclassic():
        #Parte clássica
        #Captura as tuplas (amostra, classe estanciada dpq_min)
        min_tup, max_tup, median_tup = self.sample_min_max_median()

        #Constroi circuitos quanticos com os elementos
        circs_min = build_quantum_circ(min_tup[1].elementos_iter)
        circs_max = build_quantum_circ(max_tup[1].elementos_iter)
        circs_median = build_quantum_circ(median_tup[1].elementos_iter)

        #Executa circuitos quanticos
        results_min, results_max, results_median = exec_quantum([circs_min, circs_max, circs_median])

        #Resultados do regressor
        reg_min = reg_qtmvscls(results_min, min_tup[0])
        reg_max = reg_qtmvscls(results_max, max_tup[0])
        reg_median = reg_qtmvscls(results_median, median_tup[0])
        #Cria gráficos
        graph_accvsshots_maxminmedian(self.name, reg_min, reg_max, reg_median)

        return
    
    ##-----------------
    def make_results(self, k = 10000):
        #Hyperparametros para testes:
        #Controla o numero de arvore de decisões que vão ser usadas.
        #Muitas àrvores pode gerar overfitting, mas isso é dificil de ocorrer com extra-trees
        #n_trees = [10, 50, 100, 500, 1000, 5000]
        #O numero maximo de feature que pode ser considerado para a repartição de cada nó, parametro mais importante do modelo.
        #max_features= [1,2,3,4,5,6,7,8,9,10,11,12]
        t0 = perf_counter()
        #Se existir da load e retorna falso, caso contrário um novo frame será criado.
        if(self.check_tabela()):
            self.criaDataFrame()
            
        model, X_train, X_test, y_train, y_test = self.set_model()

        y_train_pred = model.predict(X_train)
        self.make_report("Treino", y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        self.make_report("Teste", y_test, y_test_pred)
        t1 = perf_counter()
        
        self.make_speed(subname = "Criação das tabelas e treinamento do modelo", t0 = t0, t1 = t1)
        
        self.saveDataFrame()
        self.save_Y(y_test, y_test_pred, len(y_test))
        
        t0 = perf_counter()
        self.test_aleatorio(k)
        t1 = perf_counter()
        self.make_speed(subname = "teste aleatorio",t0 = t0, t1 = t1)
        
        t0 = perf_counter()
        self.make_report_ox()
        t1 = perf_counter()
        self.make_speed(subname = "Relatório observáveis:",t0 = t0, t1 = t1)
        
        t0 = perf_counter()
        self.graph_quantumvsclassic()
        t1 = perf_counter()
        self.make_speed(subname = "Parte quântica:",t0 = t0, t1 = t1)
        
        return