from numpy import int8
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import base64
import copy
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from pyairtable import Table
import os
from grid import render_grid
import matplotlib.pyplot as plt
import seaborn as sns
import ast
#import keys


st.set_page_config(page_title='Competencia', layout="wide", page_icon="⚡")


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    '''
    Desencodear archivos bin
    -----
    Args:
    > bin_file: archivo a desencodear
    -----
    Output:
    > Archivo desencodeado
    '''
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    '''
    Setear un png como background
    -----
    Args:
    > png_file: imagen
    '''
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

    
    
class Builder():
    
    '''
    Clase constructora del GUI de competencias
    '''
    
    def __init__(self):
        self.key = os.getenv('airtableKey')
        self.base_id = os.getenv('base_id')
        self.leaderboard_id = os.getenv('leaderboard_id')
        self.ytrue_id = os.getenv('ytrue_id')
        self.users_id = os.getenv('users_id')
#         self.key = keys.airtableKey
#         self.base_id = keys.base_id
#         self.leaderboard_id = keys.leaderboard_id
#         self.ytrue_id = keys.ytrue_id
#         self.users_id = keys.users_id
        self.scoring_f1 = None
        
        
    def connect(self):
        '''
        Realizar la conección a la base de airtable
        '''
        self.table = Table(api_key=self.key, base_id=self.base_id, table_name= self.leaderboard_id)
        self.y_true = Table(api_key=self.key, base_id=self.base_id, table_name=self.ytrue_id)
        self.users = Table(api_key=self.key, base_id=self.base_id, table_name=self.users_id)
        
        
        

    def load_data(self, reload):
        '''
        Cargar la data que se encuentra en la base de airtable
        -----
        Args:
        > reload: si se setea a True, solo carga el leaderboard. Si se
        setea a False, carga el y_true y usuarios también
        '''
        records = self.table.all()
        leaderboard = pd.DataFrame.from_records((r['fields'] for r in records))
        leaderboard['date'] = pd.to_datetime(leaderboard['date'], format="%Y-%m-%dT%H:%M:%S.%fZ")
        leaderboard = leaderboard.sort_values(by = ["score", "date"], ascending=[False, True])
        leaderboard['ranking'] = leaderboard["score"].rank(ascending=False, method='first').astype(int)
        self.leaderboard = leaderboard
        
        
        if reload == False:

            y_true = Table(api_key=self.key, base_id=self.base_id, table_name=self.ytrue_id)

            temp = y_true.all(fields='LABELS')
            temp = ast.literal_eval(temp[0]['fields']['LABELS'])
            y_true_df = pd.DataFrame(z.items()).set_index(0)
            y_true_df.columns = ['LABELS']
            y_true_df['LABELS'] = y_true_df['LABELS'].astype(np.int8)
            self.y_true_df = y_true_df

            shape_submit = (len(y_true_df['LABELS']), 1)
            self.shape_submit = shape_submit

            records = self.users.all()
            usuarios = pd.DataFrame.from_records((r['fields'] for r in records))
            usuarios = usuarios.set_index('indice')
            self.usuarios = usuarios
            
        else:
            pass

    
    
    def submit(self, uploaded_file, file_name, user_name):
        '''
        Submitir el archivo cargado para ver scoring
        -----
        Args:
        > uploaded_file: archivo a sumitir
        > file_name: nombre que se le da al archivo
        > user_name: nombre del usuario que realizó el submit
        '''
        try:
            dataframe = pd.read_json(uploaded_file, header=None, names=['LABELS'], dtype={'LABELS': np.int8})
        except:
            st.write('Che el formato no está bien, tiene que ser solo una tira de 1 y 0 sin nombre')
            
        date = datetime.now()
        submit_name_date = user_name + '_' + file_name + '_' + str(date.strftime("%Y_%m_%d-%I:%M:%S_%p"))

        if dataframe.shape == self.shape_submit:
            last_best_overal_f1 = self.leaderboard.score.max()
            
            dataframe['FECHA'] = date
            dataframe['SUBMITION_NAME'] = submit_name_date

            scoring_f1         = f1_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='macro')
            scoring_recall     = recall_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='macro')
            scoring_precision  = precision_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='macro')
            scoring_roc        = roc_auc_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='macro')
            
            delta = scoring_f1 - last_best_overal_f1
            
            cf_matrix = confusion_matrix(self.y_true_df['LABELS'],
                                  dataframe['LABELS'])
            
            
            if len(file_name) == 0:
                 file_name = uploaded_file.name
            submition_data = {'user_name': user_name,
                              'file_name': file_name,
                              'date': date,
                              'submit_name_date': submit_name_date,
                              'score': round(scoring_f1,3),
                              'real_filename': uploaded_file.name
                              }
            submition_data['date'] = submition_data.get('date').isoformat()
            submition_data['preds'] = str(dataframe['LABELS'].to_dict())
            #Mandamos la información a airtable
            self.table.create(submition_data)
            #Mostramos KPIs del submission
            self.kpis(scoring_f1, scoring_recall, scoring_precision, scoring_roc, delta, cf_matrix)

        else:
            st.write(f'Che el tamaño no está bien, es {dataframe.shape} cuando tiene que ser {self.shape_submit}')
            st.write(dataframe['LABELS'])
            
            
    def gui(self):    
        '''
        Función orquestadora
        '''
        #Load data
        self.connect()
        self.load_data(reload=False)
        if 'leaderboard' not in st.session_state:
            st.session_state['leaderboard'] = copy.deepcopy(self.leaderboard)
            
        st.subheader('Leaderboard')
        #grid display placeholder
        placeholder = st.empty()
        
        with st.form("submit",clear_on_submit=True):
            col1, col2 = st.columns(2)
            user_name = col1.selectbox('Quién sos?', self.usuarios)
            file_name = col2.text_input('El nombre que quieras darle')
            uploaded_file = st.file_uploader("Choose a file")
            submitted = st.form_submit_button("Evaluar")
            if submitted:
                self.submit(uploaded_file, file_name, user_name)
                self.load_data(reload=True)
                st.session_state['leaderboard'] = self.leaderboard
            
        #Show grid
        with placeholder.container():
                render_grid(st.session_state['leaderboard'])
    
    
    def kpis(self, scoring_f1, scoring_recall, scoring_precision, scoring_roc, delta, cf_matrix):
        '''
        Mostrar los KPIs de la submision
        -----
        Args:
        > scoring_f1: score del f1
        > scoring_recall: score de recall
        > scoring_precision: score de precision
        > scoring_roc: score de auc roc
        > delta: delta vs el max f1_score del leaderboard
        '''
        st.markdown('---')
        st.subheader('KPIs del submit')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 (macro)", f"{round(scoring_f1,3)}", delta=round(delta,3))
        col2.metric("Recall (macro)",f"{round(scoring_recall,3)}")
        col3.metric("Precision (macro)",f"{round(scoring_precision,3)}")
        col4.metric("ROC AUC (macro)",f"{round(scoring_roc,3)}")
        st.markdown('---')
        #Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)        
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix del submit en %\n\n');
        axes[0].set_xlabel('\nPredicted Values')
        axes[0].set_ylabel('Actual Values ');
        ## Ticket labels - List must be in alphabetical order
        axes[0].xaxis.set_ticklabels(['False','True'])
        axes[0].yaxis.set_ticklabels(['False','True'])
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d', ax=axes[1])
        axes[1].set_title('Confusion Matrix del submit en totales\n\n');
        axes[1].set_xlabel('\nPredicted Values')
        axes[1].set_ylabel('Actual Values ');
        ## Ticket labels - List must be in alphabetical order
        axes[1].xaxis.set_ticklabels(['False','True'])
        axes[1].yaxis.set_ticklabels(['False','True'])
        st.pyplot(fig)
        
        
    def run_app(self):
        '''
        Main para correr el GUI
        '''
        col1, col2, col3, col4 = st.columns(4)
        col2.image('./res/logo.png', width=550)
        set_png_as_page_bg(r"./res/background.png")
        st.markdown('---')
        self.gui()
                

if __name__ == '__main__':
    mngr = Builder()
    mngr.run_app()
