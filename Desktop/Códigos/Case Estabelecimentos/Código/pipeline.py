import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
from haversine import haversine
import geopandas as gpds
import geopy
import os
from geopy.geocoders import Nominatim
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
import csv




def do_geocode(address):
##########################
#Função para acesso ao geocode do geopy.
#Lida com exceção de TimeOut que acontece quando há mtos queries por segundo (limite de 1 por segundo)
#Retorna objeto do geocode
##########################
    geopy = Nominatim()
    try:
        return geopy.geocode(address)
    except GeocoderTimedOut:
        return do_geocode(address)

def distance(row):
    ###################
    #Função para calcular a distancia de haversine (distancia superfície esférica com base e lat/long) 
    # entre ponto de referencia CEP=01422000-> (-23.567402486525495, -46.6569533755973)
    ##################
        

    if row !=None:
        return haversine((row[0],row[1]),
                          (-23.567402486525495, -46.6569533755973))
    else:
        return np.NaN

def pre_process(df,cidade):
    ############
    #Limpeza, manipulação, seleção dos dados e geocode
    #Retorna dicionário pronto para ser enviado ao MongoDb
    ############

    df=df[['CNPJ Bas', 'CNPJ Ord', 'CNPJ DV', 'Identificador','Situaçao Cadastral', 'Data Sit.Cad.', 'Motivo Sit.Cad.','Data de Inicio', 'CNAE Principal',
       'CNAESeg','Tipo de Logradouro','Logradouro', 'Numero',
       'Complemento', 'Bairro', 'CEP', 'UF', 'Municipio']]
    df.loc[:,'Data de Inicio']=pd.to_datetime(df['Data de Inicio'],format='%Y\%m\%d')
    df.loc[:,'Ano de Inicio']=pd.DatetimeIndex(df['Data de Inicio']).year
    df.loc[:,'Mes de Inicio']=pd.DatetimeIndex(df['Data de Inicio']).month
    df.loc[:,'Dia de Inicio']=pd.DatetimeIndex(df['Data de Inicio']).day  
    df=df.merge(cidade,left_on='Municipio',right_on='Codigo',copy=False)
    #Replace Nan's
    df.loc[:,['Tipo de Logradouro','CNAESeg','Logradouro','UF','Cidade','Numero']]=df[['Tipo de Logradouro','CNAESeg','Logradouro','UF','Cidade','Numero']].replace(np.nan, '', regex=True)
    df.loc[:,['Tipo de Logradouro','CNAESeg','Logradouro','UF','Cidade','Numero']]=df[['Tipo de Logradouro','CNAESeg','Logradouro','UF','Cidade','Numero']].replace('NaN', '', regex=True)
    
    ###########
    ##Geocode##
    ###########

    df.loc[:,'Logradouro agg']=df[['Tipo de Logradouro','Logradouro']].agg(' '.join, axis=1)
    
    df.loc[:,'Endereco']=df[
    ['UF','Cidade','Bairro','Logradouro agg']].astype(str).agg(', '.join, axis=1)

    #############################
    ## Lista com endereços unicos (reduzir o numero de queries ao geocode para não ultrapassar limite) com
    ## CEP's entre [01000,01599], pois esses são da região Central de SP, onde fica o CEP de referencia
    ##################################

    unique_adrs=(df[df['CEP'].str[:5].apply(lambda x: True if str(x)<'01599' 
                                            and str(x)>'01000' else False)]).drop_duplicates('Logradouro')

    

    geolocator = Nominatim(user_agent="usuario_1")                                            
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    unique_adrs['location']=unique_adrs['Endereco'].apply(geocode)
    unique_adrs['point']=unique_adrs['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    
    unique_adrs.loc[:,'Distancia do ponto']=unique_adrs['point'].apply(distance)    

    
      
    df=df.merge(unique_adrs[['Endereco','Distancia do ponto']],how='outer',on='Endereco',copy=False,validate="m:1")                                
    ### Dropar colunas para reduzir tamanho dos dados enviados ao AtlasMongo
    df.drop(['Data de Inicio','Endereco','Logradouro agg'],axis=1,inplace=True)
    
    return df.to_dict('records')

    

def db_queries(collection,db,db_string):
    #####################################
    # Queries a serem feitas no Atlasongo em cada chunk 
    #########################################

    resultados=db['Intermediarias']
    #1
    cursor_sit_cadastral=collection.find({"Situaçao Cadastral": "02"})
    percentual_active=cursor_sit_cadastral.count()
    total_rows=collection.find({}).count()

    resultados.insert({ 
                        'Ativos':str(percentual_active),
                        'Numero total chunk': str(total_rows)})
    
        
    #2
    cursor_restaurante = collection.aggregate(
    [
        {
            '$match': {'CNAE Principal':{'$regex':'^561\d*'}}
             }

        ,
    {
        '$group': {
            '_id': '$Ano de Inicio', 
            'total': {
                '$sum': 1
            }
        }
    },
    
    { '$project': {  
      '_id': 0,
      'Ano':'$_id',
      'total': 1
   }
},
    {'$merge': {'into':{ 'db':str(db_string),'coll': 'Intermediarias'} }}
])

    #3
    collection.aggregate(
        [
    {
        '$match': {
            'Distancia do ponto': {
                '$lt': 5
            }
        }
    }, {
        '$group': {
            '_id': None, 
            '< 5km': {
                '$sum': 1
            }
        }
    },
    {
      '$project': {  
      '_id': 0,
      '< 5km':'$< 5km'

    }},
    {
        '$merge': {
            'into': {
                'db': str(db_string), 
                'coll': 'Intermediarias'
            }
        }
    }
        ])
    

    #4
    collection.aggregate(
        [
    {
        '$project': {
            '_id': '$CNAE Principal', 
            'CNAESeg': {
                '$convert': {
                    'input': '$CNAESeg', 
                    'to': 'string'
                }
            }
        }
    }, {
        '$project': {
            '_id':0,
            'Cnae1':'$_id', 
            'CNAESeg': {
                '$split': [
                    '$CNAESeg', ','
                ]
            }
        }
    }, {
        '$unwind': {
            'path': '$CNAESeg', 
            'preserveNullAndEmptyArrays': False
        }
    }
, {
        '$merge': {
            'into': {
                'db': str(db_string), 
                'coll': 'tabelaCNAE'
            }
        }
    }
]
    )
   


def queries_final(db,db_string):
    ####################
    ## Queries finais a serem executadas nos resultados finais de todos os chunks
    ####################
    resultados=db['Intermediarias']
    cursor_sit_cadastral=resultados.aggregate([
    {
        '$group': {
            '_id': None, 
            'total1': {
                '$sum': {
                    '$toInt': '$Ativos'
                }
            }, 
            'totaln': {
                '$sum': {
                    '$toInt': '$Numero total chunk'
                }
            }
        }
    }, {
        '$set': {
            'Porcentagem Ativos': {
                '$divide': [
                    '$total1', '$totaln'
                ]
            }
        }
    }
     ,{
         '$out': { 
             'db':str(db_string),'coll': 'Finais'
             } 
            }
         ])

    resultados.aggregate([
    {
        '$match': {
            'Ano': {
                '$exists': True, 
                '$ne': None
            }
        }
    }, {
        '$group': {
            '_id': '$Ano', 
            'total': {
                '$sum': '$total'
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'Ano': '$_id', 
            'Total por Ano': '$total'
        }
    }, {
        '$merge': {
            'into': {
                'db': str(db_string), 
                'coll': 'Finais'
            }
        }
    }
])

    resultados.aggregate(
    [
        {
            '$group':{
                '_id': None,
                'Total < 5km': {
                    '$sum': {
                        '$toInt': '$< 5km'
                    }
                }

            }
        }
     ,
        {
            '$project':{
                '_id':0,
                'Total < 5km':'$Total < 5km'
            }
        },

     {'$merge' : {'into':{'db':str(db_string),'coll':'Finais'}}}
    ])

    db['tabelaCNAE'].aggregate([
        {
        '$group': {
            '_id': {
                'Cnae1': '$Cnae1', 
                'Cnae2': '$CNAESeg'
            }, 
            'total': {
                '$sum': 1
            }
        }
    },
    {
        '$project':{
            '_id':0,
            'Cnae1': '$_id.Cnae1', 
            'Cnae2': '$_id.Cnae2',
            'total': '$total'

        }
    },
    {
        '$out': {'db':str(db_string),'coll':'FinalCnae'}
    }
    ]
    )
    

def export(coll_finais,coll_cnae):
    ###############
    # Exportar para CSV e EXCEl
    ##############
    os.chdir('/home/andrius/Downloads/Finais')

    #1,2,3

    df=pd.DataFrame.from_dict(coll_finais.find())

    df_ano=df.set_index('Ano', inplace=False)
    df_ano=df_ano.drop(['_id','total1','totaln','Porcentagem Ativos','Total < 5km'],axis=1)
    df_ano.dropna(inplace=True)
    df_ano.to_csv('TabelaAno.csv')

    

    
    header=['Porcentagem Ativos','Total < 5km']
    data=[df['Porcentagem Ativos'].dropna().values[0],df['Total < 5km'].dropna().values[0]]

    
    with open('./Finais.csv', 'w', encoding='UTF8') as f: 
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
    #4
    df=pd.DataFrame.from_dict(coll_cnae.find())
    df_1=df.drop(['_id'],axis=1)
    p1=df_1.pivot_table(index=['Cnae1','Cnae2'],aggfunc="sum", fill_value=0, values='total')
    p1.to_csv('TabelaCNAE.csv')  


    ##Passando para um arquivo excel
    d1=pd.read_csv('./Finais.csv')
    
    

    with pd.ExcelWriter('TabelaFinal.xlsx') as writer:  

        d1.to_excel(writer,sheet_name='DadosAtivos')
        df_ano.to_excel(writer,sheet_name='TabelaAnos')
        p1.to_excel(writer,sheet_name='TabelaCNAE')

