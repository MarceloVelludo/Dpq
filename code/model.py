import pickle
import numpy as np
import pandas as pd
from graphics import plotGraph, gen_acc_measures, quantum_report, error_Vs_j12, error_dist, feature_importance
from fpdf import FPDF
from utils import make_dir
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from check_stage import check_model
from time import perf_counter

def init_pdf(pdf, title):
    
    #pdf.add_page()
    #Font
    #Line break
    pdf.ln(5)
    pdf.set_font("Times", 'B', 13)
    #Title
    self.pdf.write(5,title)
    #pdf.cell(60,140, title,0,1, 'L')
    #Line break
    pdf.ln(1)
    pdf.set_font("Times", '', 12)
    
    return pdf

class Md:

    
    def __init__(self, df, name, k= 10000, pdf = FPDF()):
            self.name = name 
            self.path = make_dir("%s/model"%str(name))
            self.dataset = df
            self.model = None
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.k = k
            self.pdf = init_pdf(pdf,'Report modelagem: ' + self.name)
    
    def set_textimg_pdf(self, nome , descricao = '', path =''):
        self.pdf.write(5,nome)
        self.pdf.write(5,descricao)
        if path != '':
            self.pdf.cell(5,5, '',0,1, 'L')
            self.pdf.image(path,x=5,w=120,h=80)
            self.pdf.cell(5,5, '',0,1, 'L')
            
        return
    def set_textimg2_pdf(self, nome , descricao = '', path =''):
	self.pdf.write(5,nome)
   	self.pdf.write(5,descricao)
    	if path != '':
            self.pdf.cell(5,5, '',0,1, 'L')
            self.pdf.image(path,x=5,w=120,h=160)
            self.pdf.cell(5,5, '',0,1, 'L')                               

    	return
    
    ### modelo  
    def set_model(self):
            #Se o modelo não existir faz um novo e guarda ele.
            if check_model(self.name):
                t1 = perf_counter()
                self.model = pickle.load(open(self.path + "/trained_model", 'rb'))
                self.X_train = pickle.load(open(self.path + "/X_train", 'rb'))
                self.X_test = pickle.load(open(self.path + "/X_test", 'rb'))
                self.y_train = pickle.load(open(self.path + "/y_train", 'rb'))
                self.y_test = pickle.load(open(self.path + "/y_test", 'rb'))
                t2 = perf_counter()
                print("Loading do modelo feito em: %s segundos"%str(t2-t1))
                #Gráfico da importância das features
                t1 = perf_counter()
                self.set_textimg2_pdf(nome="Gráfico de importância das caracteristicas do modelo:" , descricao = 'Este gráfico é independente de qualquer data set.', path =feature_importance(self.dataset.columns[5:], self.model, self.name))
                t2 = perf_counter()
                print("Feature importance feito em: %s segundos"%str(t2-t1))
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset.iloc[:,5:], self.dataset.iloc[:,4], test_size=0.3, random_state= 7)
                self.model = ExtraTreesRegressor(n_estimators=100, random_state=0, n_jobs= -1).fit(self.X_train, self.y_train)
                pickle.dump(self.model, open(self.path + "/trained_model", 'wb'))
                pickle.dump(self.X_train, open(self.path + "/X_train", 'wb'))
                pickle.dump(self.X_test, open(self.path + "/X_test", 'wb'))
                pickle.dump(self.y_train , open(self.path + "/y_train", 'wb'))
                pickle.dump(self.y_test , open(self.path + "/y_test", 'wb'))
                #Gráfico de importancia das features.
                self.set_textimg2_pdf(nome="Gráfico de importância das caracteristicas do modelo:" , descricao = 'Este gráfico é independente de qualquer data set.', path =feature_importance(self.dataset.columns[5:], self.model, self.name))
            return self.model, self.X_train, self.X_test, self.y_train, self.y_test
    
    def make_report(self, name, X_train, y_real, y_pred):
        y_real = y_real
        relative_error_1 = 0
        relative_error_2 = 0
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real,y_pred)

        relative_error =[]
        for y_p,y_r  in zip(y_pred, y_real):
            if y_r !=0:
                #print("Y_r:",y_r)
                relative_error.append(np.abs((y_r-y_p)/y_r))
            else:
                relative_error.append(0)
        
        re = np.mean(relative_error)

        #String com os resultados
        rst_string = ''.join([name, "\nMédia do erro absoluto: %f \nMédia quadrada do erro: %f \nR2: %f  \nrelative_error(np.abs((y_t-y_p)/y_t)/n): %f\n" % (mae, mse, r2, re)])       
        self.set_textimg_pdf("Resultados do modelo ML:\n", rst_string)
        #print("\nRESULTADOS:", rst_string)
        #Graficos
        self.set_textimg_pdf('Gráficos modelo Extra Trees Regressor para: %s'%name, '',plotGraph(y_real, y_pred, "Extra Trees Regressor - %s "%name, self.name, mae=mae, mse=mse, r2=r2))        
        self.set_textimg_pdf('', '',gen_acc_measures(pd.DataFrame(X_train),y_real, self.name, name))
        
        
        self.set_textimg_pdf('Gráfico distribuição do valor dos erros:%s'%name,'Gráfico que mostra a frequencia do erro pelo seu valor.Um Bom modelo terá a distribuição parecida com a normal.',path=error_dist(y_real, y_pred, name, self.name,num_bins = 20))
        self.set_textimg_pdf("Grafico da distribuição do erro por j12%s"%name, "Este gráfico visa localizar pontos de J12 onde se tem mais erro. O ideal é que estes pontos não ocorram e ele tenha um erro mais uniforme por todo j12", error_Vs_j12(y_real, y_pred, name, self.name,intervalo=0.2))

        return [mae, mse, r2]
    
    def model_results(self, df_ale):
        
        y_train_pred = self.model.predict(self.X_train)
        self.make_report(name="Treino",X_train = self.X_train, y_real = self.y_train, y_pred = y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        self.make_report("Teste", self.X_test,self.y_test, y_test_pred)
        
        self.teste_aleatorio(df_ale)
        
        return 
    
    #Teste aleatório com base de dados de 10k
    def teste_aleatorio(self, df):

        X_val = df.iloc[:,5:]
        y_val = df.iloc[:,4]
        y_val_pred = self.model.predict(X_val)
        self.make_report("Validação", X_val,y_val, y_val_pred)

        return

    
    def output(self):
        self.pdf.output("../experimentos/%s/RelatórioFinal.pdf"%self.name, "F")
        return

            
    
