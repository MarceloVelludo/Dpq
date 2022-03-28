import numpy as np
import pandas as pd
from fpdf import FPDF
from check_stage import check_dataframe
from decimal import Decimal
from sympy.physics.quantum import TensorProduct
from itertools import product
from time import perf_counter
from scipy.linalg import expm
from utils import make_dir
from graphics import criaGraficos
import random

def init_pdf(pdf, title):
    try:
        #Font
        pdf.set_font("Times", 'B', 15)
        #Title
        self.pdf.write(5,title)
        #pdf.cell(60,140, title,1,0, 'L')
        #Line break
        pdf.ln(1)
        pdf.set_font("Times", '', 12)
    except:
        pdf.add_page()
        #Font
        pdf.set_font("Times", 'B', 15)
        #Title
        self.pdf.write(5,title)
        #pdf.cell(60,140, title,1,0, 'L')
        #Line break
        pdf.ln(1)
        pdf.set_font("Times", '', 12)
    
    return pdf

# Classe Eq_cl guarda as funções e realiza os calculos com o computador clássico.
class Eq_cl:

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
                 tInicial=1, tFinal=20, passoT=1, pdf = FPDF('P', 'mm', 'A4')):
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
        
        self.dataset = None
        
        #Para criar relatorios
        #Nome versão mais compacta e nome completo 
        self.name = str("["+str(self.j_1_inicial)[:4]+":"+str(self.j_1_final)[:4]+":"+str(self.passoJ_1)[:4]+"]"+"["+str(self.j_2_inicial)[:4]+":"+str(self.j_2_final)[:4]+":"+str(self.passoJ_2)[:4]+"]"+"["+str(self.bz_1_inicial)[:4]+":"+str(self.bz_1_final)[:4]+":"+str(self.passoBz_1)[:4]+"]"+"["+str(self.bz_2_inicial)[:4]+":"+str(self.bz_2_final)[:4]+":"+str(self.passoBz_2)[:4]+"]"+"["+str(self.j_12_inicial)[:4]+":"+str(self.j_12_final)[:4]+":"+str(self.passoJ_12)[:4]+"]"+"["+str(self.tInicial)[:3]+":"+str(self.tFinal)[:3]+":"+str(self.passoT)[:4]+"]")
        #Cria diretório para guardar tabelas
        make_dir(self.name)
        make_dir(self.name+"/tabelas")
        #Cria pagina pdf para relatorio
        self.pdf = init_pdf(pdf,'Relatorio para os parametros: \n' + self.name +' [Inicio:Término:Passo] \n j1,j2,bz1,bz2,j12,tempo')
        self.pdf.cell(3,20, "Tamanho da lista de elementos:"+str(len(self.elementos_iter)),0,1, 'L')
    
    def set_textimg_pdf(self, nome , descricao = '', path =''):
        self.pdf.write(5,nome)
        self.pdf.write(5,descricao)
        if path != '':
            self.pdf.cell(5,5, '',0,1, 'L')
            self.pdf.image(path,x=5,w=120,h=80)
            self.pdf.cell(5,5, '',0,1, 'L')                               

        return
        

        
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
    
    #Cria data frame com os parametros da classe.
    def criaDataFrame(self, complemento_nome = ""): 
        if check_dataframe(self.name,complemento_nome):
            self.dataset = self.loadDataFrame(path = "../experimentos/"+self.name+"/tabelas/"+complemento_nome+ self.name + ".csv")
        else:
            self.dataset = pd.DataFrame(self.criaFrameNovo(), columns = self.getNames())
            self.saveDataFrame(complemento_nome)
        return self.dataset
    
    def loadDataFrame(self, path):
        self.dataset = pd.read_csv(path, index_col = 0)
        return self.dataset
    
    def saveDataFrame(self, complemento_nome = ""):
        if self.dataset is None:
            self.criaDataFrame()
        self.dataset.to_csv(path_or_buf="../experimentos/"+self.name+"/tabelas/"+ self.name + complemento_nome+".csv")
        return
    
    def saveDataFrameY(self, datasetY):
        datasetY.to_csv(path_or_buf="../experimentos/"+self.name+"/tabelas/yRealyPred-"+ self.name + ".csv")
        return
    
    def save_Y(self, yReal, yPred, lenght):
        #print("shape:", np.array(yReal).reshape((lenght,-1)).shape)
        yReal = np.array(yReal).reshape((lenght,-1))
        yPred = np.array(yPred).reshape((lenght,-1))
        datasetY = pd.DataFrame(np.hstack([yReal, yPred]), columns = ["yReal","yPred"])
        self.saveDataFrameY(datasetY)
        return datasetY
        #Função que retorna samples para os minimos, máximos e mediana
    def sample_min_max_median(self,n=0):
        #Cria tabelas com minimo maximo e médiana.
        #Minimos
        dpq_min = Eq_cl(j_1_inicial= self.j_1_inicial, j_1_final= self.j_1_inicial+self.passoJ_1*n, passoJ_1 = self.passoJ_1,
                                          j_2_inicial= self.j_2_inicial, j_2_final= self.j_2_inicial+self.passoJ_2*n, passoJ_2 = self.passoJ_2,
                                          bz_1_inicial= 0, bz_1_final= 0, passoBz_1 = self.passoBz_1,
                                          bz_2_inicial= 0, bz_2_final= 0, passoBz_2 = self.passoBz_2,
                                          j_12_inicial= self.j_12_inicial, j_12_final= self.j_12_inicial+self.passoJ_12*n, passoJ_12 = self.passoJ_12,
                                          tInicial=self.tInicial, tFinal=self.tFinal, passoT=self.passoT)
        sample_min = dpq_min.criaDataFrame()

        #Máximos
        dpq_max = Eq_cl(j_1_inicial= self.j_1_final-self.passoJ_1*n, j_1_final= self.j_1_final, passoJ_1 = self.passoJ_1,
                                          j_2_inicial= self.j_2_final-self.passoJ_2*n, j_2_final= self.j_2_final, passoJ_2 = self.passoJ_2,
                                          bz_1_inicial= 0, bz_1_final= 0, passoBz_1 = self.passoBz_1,
                                          bz_2_inicial= 0, bz_2_final= 0, passoBz_2 = self.passoBz_2,
                                          j_12_inicial= self.j_12_final-self.passoJ_12*n, j_12_final= self.j_12_final, passoJ_12 = self.passoJ_12,
                                          tInicial=self.tInicial, tFinal=self.tFinal, passoT=self.passoT)
        sample_max = dpq_max.criaDataFrame()

        #Mediana
        dpq_median = Eq_cl(j_1_inicial= ((self.j_1_inicial+self.j_1_final)/2) - ((self.passoJ_1*n)/2), j_1_final=((self.j_1_inicial+self.j_1_final)/2) + ((self.passoJ_1*n)/2) , passoJ_1 = self.passoJ_1,
                                          j_2_inicial= ((self.j_2_inicial+self.j_2_final)/2) - ((self.passoJ_2*n)/2), j_2_final= ((self.j_2_inicial+self.j_2_final)/2) + ((self.passoJ_2*n)/2), passoJ_2 = self.passoJ_2,
                                          bz_1_inicial= 0, bz_1_final= 0, passoBz_1 = self.passoBz_1,
                                          bz_2_inicial= 0, bz_2_final= 0, passoBz_2 = self.passoBz_2,
                                          j_12_inicial= ((self.j_12_inicial+self.j_12_final)/2) - ((self.passoJ_12*n)/2), j_12_final= ((self.j_12_inicial+self.j_12_final)/2) + ((self.passoJ_12*n)/2), passoJ_12 = self.passoJ_12,
                                          tInicial=self.tInicial, tFinal=self.tFinal, passoT=self.passoT)
        sample_median = dpq_median.criaDataFrame()

        return (sample_min, dpq_min), (sample_max, dpq_max), (sample_median, dpq_median)
    
    def test_aleatorio(self, k= 10000):
        path = "../experimentos/%s/tabelas/%sk-%s.csv"%(self.name, self.name,str(k))
        if check_dataframe(self.name, "k-%s"%str(k)):
            self.df = self.loadDataFrame(path)
        
        else:
            
            arrayJ_1 = self.random_samples(self.j_1_inicial, self.j_2_final, 10)
            arrayJ_2 = self.random_samples(self.j_2_inicial, self.j_2_final, 10)
            arrayBz_1 = self.random_samples(self.bz_1_inicial, self.bz_1_final, 10)
            arrayBz_2 = self.random_samples(self.bz_2_inicial, self.bz_2_final, 10)
            arrayJ_12 = self.random_samples(self.j_12_inicial, self.j_12_final, 10)

            self.elementos_iter = random.sample(list(product(arrayJ_1, arrayJ_2, arrayBz_1, arrayBz_2,  arrayJ_12)), k)
            #print("\n len teste aleatorio:", len(self.elementos_iter))
            self.df = self.criaDataFrame("k-%s"%str(k))
            
        return self.df
    
    ###Teste Aleatório
    def random_float(self,low, high):
         return random.random()*(high-low) + low

    def random_samples(self,low, high, k):
        samples = []
        for number_samples in range(k):
            samples.append(self.random_float(low,high))
         
        return samples
    
    #Cria frame para observaveis.
    def criaFrameGraficos(self):
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

        return pd.DataFrame(ox1, columns = ['ox1']), pd.DataFrame(ox2, columns = ['ox2']), pd.DataFrame(oy1, columns = ['oy1']), pd.DataFrame(oy2, columns = ['oy2']), pd.DataFrame(oz1, columns = ['oz1']), pd.DataFrame(oz2, columns = ['oz2']), pd.DataFrame(tempos, columns = ['tempo'])
    
    #Gera relatorio das observaveis em relação ao tempo.
    def make_report_ox(self):
        #self.set_textimg_pdf("Grafico das observaveis com relação ao tempo:", descricao = "Observaveis e o tempo nos pontos minimos, maximos e medios", path ='')
        #Para os pontos minimos
        dpqMin = Eq_cl(j_1_inicial=self.j_1_inicial, j_1_final=self.j_1_inicial, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_inicial, j_2_final=self.j_2_inicial, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_inicial, bz_1_final=self.bz_1_inicial, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_inicial, bz_2_final=self.bz_2_inicial, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_inicial, j_12_final=self.j_12_inicial, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMin.criaFrameGraficos()
        self.set_textimg_pdf('Ox1-Min', descricao = '', path =criaGraficos(ox1, tempos, dpqMin.name, "Ox1-Min"))
        self.set_textimg_pdf('Ox2-Min', descricao = '', path =criaGraficos(ox2, tempos, dpqMin.name, "Ox2-Min"))
        self.set_textimg_pdf('Oy1-Min', descricao = '', path =criaGraficos(oy1, tempos, dpqMin.name, "Oy1-Min"))
        self.set_textimg_pdf('Oy2-Min', descricao = '', path =criaGraficos(oy2, tempos, dpqMin.name, "Oy2-Min"))
        self.set_textimg_pdf('Oz1-Min', descricao = '', path =criaGraficos(oz1, tempos, dpqMin.name, "Oz1-Min"))
        self.set_textimg_pdf('Oz2-Min', descricao = '', path =criaGraficos(oz2, tempos, dpqMin.name, "Oz2-Min"))

        #Para os pontos Maximos
        dpqMax = Eq_cl(j_1_inicial=self.j_1_final, j_1_final=self.j_1_final, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_final, j_2_final=self.j_2_final, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_final, bz_1_final=self.bz_1_final, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_final, bz_2_final=self.bz_2_final, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_final, j_12_final=self.j_12_final, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMax.criaFrameGraficos()
        self.set_textimg_pdf('Ox1-Max', descricao = '', path =criaGraficos(ox1, tempos, dpqMax.name, "Ox1-Max"))
        self.set_textimg_pdf('Ox2-Max', descricao = '', path =criaGraficos(ox2, tempos, dpqMax.name, "Ox2-Max"))
        self.set_textimg_pdf('Oy1-Max', descricao = '', path =criaGraficos(oy1, tempos, dpqMax.name, "Oy1-Max"))
        self.set_textimg_pdf('Oy2-Max', descricao = '', path =criaGraficos(oy2, tempos, dpqMax.name, "Oy2-Max"))
        self.set_textimg_pdf('Oz1-Max', descricao = '', path =criaGraficos(oz1, tempos, dpqMax.name, "Oz1-Max"))
        self.set_textimg_pdf('Oz2-Max', descricao = '', path =criaGraficos(oz2, tempos, dpqMax.name, "Oz2-Max"))

        #Para os pontos médios
        dpqMean = Eq_cl(j_1_inicial=self.j_1_final/2, j_1_final=self.j_1_final/2, passoJ_1 = 0.5,
                                             j_2_inicial=self.j_2_final/2, j_2_final=self.j_2_final/2, passoJ_2 = 0.5,
                                             bz_1_inicial=self.bz_1_final/2, bz_1_final=self.bz_1_final/2, passoBz_1 = 0.5,
                                             bz_2_inicial=self.bz_2_final/2, bz_2_final=self.bz_2_final/2, passoBz_2 = 0.5,
                                             j_12_inicial=self.j_12_final/2, j_12_final=self.j_12_final/2, passoJ_12 = 0.1,
                                             tInicial=1, tFinal=20, passoT=1)
        ox1, ox2, oy1, oy2, oz1, oz2, tempos = dpqMean.criaFrameGraficos()
        self.set_textimg_pdf('Ox1-Mediana', descricao = '', path =criaGraficos(ox1, tempos, dpqMean.name, "Ox1-Mediana"))
        self.set_textimg_pdf('Ox2-Mediana', descricao = '', path =criaGraficos(ox2, tempos, dpqMean.name, "Ox2-Mediana"))
        self.set_textimg_pdf('Oy1-Mediana', descricao = '', path =criaGraficos(oy1, tempos, dpqMean.name, "Oy1-Mediana"))
        self.set_textimg_pdf('Oy2-Mediana', descricao = '', path =criaGraficos(oy2, tempos, dpqMean.name, "Oy2-Mediana"))
        self.set_textimg_pdf('Oz1-Mediana', descricao = '', path =criaGraficos(oz1, tempos, dpqMean.name, "Oz1-Mediana"))
        self.set_textimg_pdf('Oz2-Mediana', descricao = '', path =criaGraficos(oz2, tempos, dpqMean.name, "Oz2-Mediana"))

        return