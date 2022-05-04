from numpy import int8
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import base64
import copy
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from pyairtable import Table
import os
from grid import render_grid



st.set_page_config(page_title='Competencia', layout="wide", page_icon="📒")


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
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
    -----
    Args:
    > 
    '''
    
    def __init__(self):
        self.key = os.getenv('airtableKey')
        self.base_id = os.getenv('base_id')
        self.leaderboard_id = os.getenv('leaderboard_id')
        self.ytrue_id = os.getenv('ytrue_id')
        self.users_id = os.getenv('users_id')
        self.scoring_f1 = None
        
        
    def connect(self):
        self.table = Table(api_key=self.key, base_id=self.base_id, table_name= self.leaderboard_id)
        self.y_true = Table(api_key=self.key, base_id=self.base_id, table_name=self.ytrue_id )
        self.users = Table(api_key=self.key, base_id=self.base_id, table_name=self.users_id)
        
        
        

    def load_data(self, reload):
        records = self.table.all()
        leaderboard = pd.DataFrame.from_records((r['fields'] for r in records))
        leaderboard['date'] = pd.to_datetime(leaderboard['date'], format="%Y-%m-%dT%H:%M:%S.%fZ")
        leaderboard = leaderboard.sort_values(by = ["score", "date"], ascending=[False, True])
        leaderboard['ranking'] = leaderboard["score"].rank(ascending=False, method='first').astype(int)
        self.leaderboard = leaderboard
        
        
        if reload == False:
            records = self.y_true.all()
            y_true_df = pd.DataFrame.from_records((r['fields'] for r in records))
            y_true_df = y_true_df.sort_values(by='indice').reset_index(drop=True)
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
        try:
            dataframe = pd.read_csv(uploaded_file, header=None, names=['LABELS'], dtype={'LABELS': np.int8})
        except:
            st.write('Che el formato no está bien, tiene que ser solo una tira de 1 y 0 sin nombre')
            
        date = datetime.now()
        submit_name_date = user_name + '_' + file_name + '_' + str(date.strftime("%Y_%m_%d-%I:%M:%S_%p"))

        if dataframe.shape == self.shape_submit:
            last_best_overal_f1 = self.leaderboard.score.max()
            
            dataframe['FECHA'] = date
            dataframe['SUBMITION_NAME'] = submit_name_date

            scoring_f1         = f1_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='weighted')
            scoring_recall     = recall_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='weighted')
            scoring_precision  = precision_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='weighted')
            scoring_roc        = roc_auc_score(self.y_true_df['LABELS'],
                                  dataframe['LABELS'], average='weighted')
            
            delta = scoring_f1 - last_best_overal_f1
            
            if len(file_name) == 0:
                 file_name = uploaded_file.name
            submition_data = {'user_name': user_name,
                              'file_name': file_name,
                              'date': date,
                              'submit_name_date': submit_name_date,
                              'score': scoring_f1,
                              'real_filename': uploaded_file.name
                              }
            submition_data['date'] = submition_data.get('date').isoformat()
            submition_data['preds'] = str(dataframe['LABELS'].to_dict())
            self.table.create(submition_data)
            self.kpis(scoring_f1, scoring_recall, scoring_precision, scoring_roc, delta)

        else:
            st.write(f'Che el tamaño no está bien, es {dataframe.shape} cuando tiene que ser {self.shape_submit}')
            st.write(dataframe['LABELS'])
            
            
    def gui(self):    
        '''
        Funtion to run rulesGUI
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
    
    
    def kpis(self, scoring_f1, scoring_recall, scoring_precision, scoring_roc, delta):
        st.markdown('---')
        st.subheader('KPIs del submit')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 (micro)", f"{round(scoring_f1,3)}", delta=round(delta,3))
        col2.metric("Recall (micro)",f"{round(scoring_recall,3)}")
        col3.metric("Precision (micro)",f"{round(scoring_precision,3)}")
        col4.metric("ROC AUC (micro)",f"{round(scoring_roc,3)}")
        st.markdown('---')
            

    def run_app(self):
        '''
        Main function to run nuqlea API
        '''
        col1, col2, col3, col4 = st.columns(4)
        col2.image('./res/logo.png', width=550)
        set_png_as_page_bg(r"./res/background.png")
        st.markdown('---')
        self.gui()
                

if __name__ == '__main__':
    mngr = Builder()
    mngr.run_app()