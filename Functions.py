import numpy as np
from qutip import *
import tensorflow.compat.v1 as tf


def verificarHermit (lista):
    return sum([lista[i][j].isherm for i in range(len(lista)) for j in range(len(lista[0]))])

#Leitores de arquivos de texto
def text_to_rho(txt,d):
    b=[]
    for cont in range(len(txt)):
        a=[]
        for j in range(len(txt[cont])):
            for item in txt[cont][j].split(','):
                s=item.replace('i','j')
                a.append(complex(s))
        b.append(np.concatenate((np.real(np.array(a)),np.imag(np.array(a)))))

    return np.array(b)

def text_to_rho_qutip(rho,d):
    b=[]
    for cont in range(len(rho)):
            b.append(Qobj((rho[cont][0:d*d]+1j*rho[cont][d*d:]).reshape(d,d)))
    return b

def dataSplit (estados,nsplit,n_steps):
    d=len(estados[0][0])/2
    #Dividindo treino/teste
    def split_test(x, nsplit):
        x_train=x[:-nsplit]
        x_test=x[-nsplit:]
        return x_train,x_test

    #Dividir cada sequencia em passos temporais
    #len(lista_train) será o número de splits que contemplam a evolução completa de um unico estado
    def split_sequence(sequence, n_steps):
        X, y , lista = [], [] ,[]
        for i in range(len(sequence)):
            # Encontrar final
            end_ix = i + n_steps
            # Conferir se chegou no final
            if end_ix >= len(sequence)-1:
                break
            #Ver eq 33/34 do slide defesa

            seq_x, seq_y = sequence[i:end_ix], (sequence[(i+1):(end_ix+1)]-sequence[i:end_ix])
            #opção 1 : seq_x, seq_y = sequence[i:end_ix], sequence[end_ix-1]-sequence[i]
            #seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:(end_ix+n_steps)]
            X.append(seq_x)
            y.append(seq_y)
            lista.append(i)
        return X,y,lista

    #tem que ser multiplo do tamanho de cada sequencia temporal
    [train,test]=split_test(estados,nsplit)

    train=np.array(train)
    test=np.array(test)
    data_dim = 2*d*d
    #Para fazer moving window
    tempx=np.zeros((len(estados[0]),len(estados[0][0])))
    tempy=np.zeros((len(estados[0]),len(estados[0][0])))
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for i in range(len(train)):
        tempx,tempy,lista_train=split_sequence(train[i],n_steps)
        x_train=x_train+tempx
        y_train=y_train+tempy

    for i in range(len(test)):
        tempx,tempy,lista_test=split_sequence(test[i],n_steps)
        x_test=x_test+tempx
        y_test=y_test+tempy
    #np.array(y).reshape(len(x),len(x)[0],len(x)[0][0])
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    y_train=np.reshape(y_train,(len(y_train),n_steps,len(y_train[0][0])))
    y_test=np.reshape(y_test,(len(y_test),n_steps,len(y_test[0][0])))
    return x_train,y_train,x_test,y_test



#Função de custo
def Z(delta_t,d):
    def J(data,y_pred):
        y_true=data[:,:,:,1]
        rho_input=data[:,:,:,0] 
        
        re,imag=tf.split(y_true,2,axis=2)
        comple=tf.complex(re,imag)

        termo3=tf.dtypes.cast(tf.reshape(comple,[tf.shape(comple)[0],tf.shape(comple)[1],d,d]),tf.complex128)#
        re,imag=tf.split(y_pred,2,axis=2)
        y_pred_comp=tf.dtypes.cast(tf.reshape((tf.complex(re,imag)),(tf.shape(y_pred)[0],tf.shape(y_pred)[1],d,d)),tf.complex128)
       
        
        re,imag=tf.split(rho_input,2,axis=2)
        comple=tf.complex(re,imag)

        L_rho=tf.matmul(y_pred_comp,tf.dtypes.cast(
                tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d)),tf.complex128))
        L_rho_L=tf.matmul(L_rho,tf.linalg.adjoint(y_pred_comp))        
    
    
        temp=tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d))
        L_L_rho=tf.matmul(tf.matmul(
                                y_pred_comp,tf.linalg.adjoint(y_pred_comp)),
                                    tf.dtypes.cast(temp,tf.complex128))      

        
        rho_L_L=tf.matmul(tf.dtypes.cast(
                                tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d)),tf.complex128),tf.matmul
                            (y_pred_comp,tf.linalg.adjoint(y_pred_comp)))
        print(L_rho_L.shape)
        return (tf.math.reduce_sum(tf.dtypes.cast(tf.linalg.norm(termo3-delta_t*
        (L_rho_L-0.5*(L_L_rho+rho_L_L)),ord='fro',axis=[-2,-1]),tf.float32)))/tf.dtypes.cast(tf.shape(L_rho_L)[0]*tf.shape(L_rho_L)[1],tf.float32)
    return J

#Função de custo
def Z_y_é_delta(A_op_t,rho_input,delta_t,d):
    def J(y_true,y_pred):  
    
        re,imag=tf.split(y_true,2,axis=2)
        comple=tf.complex(re,imag)

        termo3=tf.dtypes.cast(tf.reshape(comple,[tf.shape(comple)[0],tf.shape(comple)[1],d,d]),tf.complex128)#
        re,imag=tf.split(y_pred,2,axis=2)

        y_pred_comp=tf.reshape((tf.complex(re,imag)),(tf.shape(y_pred)[0],tf.shape(y_pred)[1],d,d))
        
       

        alpha_A=tf.tensordot(tf.dtypes.cast(A_op_t,tf.complex128),tf.dtypes.cast(y_pred_comp,tf.complex128),
                             axes=[[1],[2]])
        alpha_A=tf.transpose(alpha_A,perm=[1,2,0,3])
        alpha_A=tf.reshape(alpha_A,(tf.shape(alpha_A)[0],tf.shape(alpha_A)[1],d,d))#
        
        #tf.reshape(alpha_A,(tf.shape(alpha_A)[0],d,d))
 
        re,imag=tf.split(rho_input,2,axis=2)
        comple=tf.complex(re,imag)
        
        
      
        
        alpha_A_rho=tf.matmul(alpha_A,tf.dtypes.cast(
                tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d)),tf.complex128))
       

        termo1=tf.matmul(alpha_A_rho,tf.linalg.adjoint(alpha_A))
        comple=tf.complex(re,imag)
        comple=tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d))        
    
    
        temp=tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d))
        termo2=tf.matmul(tf.matmul(
                                alpha_A,tf.linalg.adjoint(alpha_A)),
                                    tf.dtypes.cast(temp,tf.complex128))      

        
        termo2_2=tf.matmul(tf.dtypes.cast(
                                tf.reshape(comple,(tf.shape(comple)[0],tf.shape(comple)[1],d,d)),tf.complex128),tf.matmul
                            (alpha_A,tf.linalg.adjoint(alpha_A)))
       
        return (tf.math.reduce_sum(tf.dtypes.cast(tf.linalg.norm(termo3-delta_t*(termo1-0.5*(termo2+termo2_2)),ord='fro',axis=[-2,-1]),tf.float32)))/tf.dtypes.cast(tf.shape(termo1)[0]*tf.shape(termo1)[1],tf.float32)
    return J






