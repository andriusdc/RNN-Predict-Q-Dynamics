import numpy as np
import pandas as pd
from qutip import *
import tensorflow.compat.v1 as tf

class ModeloPredicao():
    def __init__(self,**kwargs):
        #estados_getgama,estados_test,timestamps,model,timestep,step_index,divid

        model2=tf.keras.models.clone_model(kwargs['model'])
        model2.set_weights(kwargs['model'].get_weights())   
        self.model2=model2
        self.estados_getgama=kwargs['estados_getgama']
        self.step_index=kwargs['step_index']
        self.divid=kwargs['divid']
        self.d=int(self.divid/2)
        self.timestep=kwargs['timestep']
        self.data_dim=self.divid*2
        self.estados_test=kwargs['estados_test']
        self.timestamps=kwargs['timestamps']
           
    def getgama(self):
        prediçao_complex=np.zeros((len(self.estados_getgama),self.step_index,self.d*self.d),dtype=complex)
        prediçao_modelo=np.zeros((len(self.estados_getgama),self.step_index,2*self.d*self.d),dtype=complex)  
        prediçao_qobj=np.zeros((len(self.estados_getgama),self.step_index,self.d,self.d),dtype=complex)
        gama=np.zeros((self.step_index,self.d*self.d),dtype=complex)


        for a in range(len(self.estados_getgama)):
            self.model2.reset_states()
            for j in range(self.step_index):    
                prediçao_modelo[a][j]=self.model2.predict(self.estados_getgama[a][j].reshape(1,1,self.data_dim),batch_size=1)  
                prediçao_complex[a][j]=prediçao_modelo[a][j][0:self.divid]+1j*prediçao_modelo[a][j][self.divid:]
                gama[j]=np.sum(prediçao_complex[:,j], axis=0)/len(prediçao_complex) 
                prediçao_qobj[a][j]=Qobj(prediçao_complex[a][j].reshape(self.d,self.d))
                
        self.gama=gama
        gama_mediatotal=np.sum(prediçao_complex, axis=(0,1))/(len(prediçao_complex[0])*len(prediçao_complex))
        
        self.prediçao_qobj=prediçao_qobj
        self.prediçao_complex=prediçao_complex  
        return gama_mediatotal,prediçao_qobj,prediçao_complex

    def getstates(self):
        H=Qobj(np.zeros((self.d,self.d)))
        AlphaA=np.matrix(np.dot(np.array(qeye(self.d)),self.gama_mediatotal.reshape(self.d,self.d))) 
        prediçao_qutip=[mesolve(H,Qobj((self.estados_test[i][0][0:self.divid]+1j*self.estados_test[i][0][self.divid:]).reshape(self.d,self.d)),self.timestamps,Qobj(AlphaA))for i in range(len(self.estados_test))]  
        estados_test_qutip=np.zeros((len(self.estados_test),self.timestep,self.d,self.d),dtype=complex)
        fid_final=np.zeros((len(self.estados_test),self.timestep))
        tracedist_final=np.zeros((len(self.estados_test),self.timestep)) 
        for a in range(len(self.estados_test)):
            norm=[]
            fid=[] 
            for t in range(self.timestep):
                estados_test_qutip[a][t]=(self.estados_test[a][t][0:self.divid]+1j*self.estados_test[a][t][self.divid:]).reshape(self.d,self.d)
                #Distancias entre estado previsto com gama medio e estado ideal
                fid.append(fidelity(Qobj(estados_test_qutip[a][t]),Qobj(prediçao_qutip[a].states[t])))
                norm.append(tracedist(Qobj(estados_test_qutip[a][t]),Qobj(prediçao_qutip[a].states[t])))
            fid_final[a]=fid
            tracedist_final[a]=norm
        lista=[]
        for i in range(len(self.estados_test)):
            lista.append("Estado"+str(i))
        lista2=[]
        for i in range(len(self.estados_test)):
            lista2.append("Estado"+str(i))
        #Gama médio por instante de tempo    
        media_fid=np.zeros((self.timestep))
        media_trace=np.zeros((self.timestep))

        norm_df = pd.DataFrame(np.swapaxes(tracedist_final,0,1),columns=lista)
        fid_df = pd.DataFrame(np.swapaxes(fid_final,0,1),columns=lista)

        return fid_df,norm_df,prediçao_qutip
    def predicao(self):
        self.gama_mediatotal,self.prediçao_qobj,self.prediçao_complex=self.getgama()
        
        return self.getstates() 

    

        

    
         




