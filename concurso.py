from numpy import int8
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from pyairtable import Table
import os
import seaborn as sns

api_key = os.getenv('AIRTABLE_API_KEY')

table = Table(api_key=api_key, base_id='app89yIsvduofuVNb', table_name='tbl6krugpB1Uvlcpu')

y_true = Table(api_key=api_key, base_id='app89yIsvduofuVNb', table_name='tblXYnOlzZ6yT5eCo')

dtypes_leaderboard = {"user_name": "object",
                      "file_name": "object",
                      "submit_name_date": "object",
                      "score": "float64",
                      "ranking": "int64",
                      "original_filename": "object"}

records = table.all()
leaderboard = pd.DataFrame.from_records((r['fields'] for r in records))
leaderboard['date'] = pd.to_datetime(leaderboard['date'], format="%Y-%m-%dT%H:%M:%S.%fZ")
leaderboard = leaderboard.sort_values(by = ["score", "date"], ascending=[False, True])
leaderboard['ranking'] = leaderboard["score"].rank(ascending=False, method='first').astype(int)


st.header('Este es el ranking actual')
st.write(leaderboard[['ranking', 'user_name', 'file_name', 'date', 'score']])

records = y_true.all()
y_true_df = pd.DataFrame.from_records((r['fields'] for r in records))
y_true_df = y_true_df.sort_values(by='indice').reset_index(drop=True)
y_true_df['LABELS'] = y_true_df['LABELS'].astype(np.int8)

shape_submit = (len(y_true_df['LABELS']), 1)

user_name = st.text_input('Tu nombre')
file_name = st.text_input('El nombre que quieras darle')

date = datetime.now()
submit_name_date = user_name + '_' + file_name + '_' + str(date.strftime("%Y_%m_%d-%I:%M:%S_%p"))
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:

    try:
        dataframe = pd.read_csv(uploaded_file, header=None, names=['LABELS'], dtype={'LABELS': np.int8})

        if dataframe.shape == shape_submit:

            dataframe['FECHA'] = date
            dataframe['SUBMITION_NAME'] = submit_name_date

            if st.button('Evaluar', key='scoring'):

                scoring_f1 = f1_score(y_true_df['LABELS'],
                                      dataframe['LABELS'])

                st.write('Su submit tiene un F1 de: ')
                st.write(scoring_f1)

                if len(file_name) == 0:
                     file_name = uploaded_file.name

                submition_data = {'user_name': user_name,
                                  'file_name': file_name,
                                  'date': date,
                                  'submit_name_date': submit_name_date,
                                  'score': scoring_f1,
                                  'real_filename': uploaded_file.name
                                  }

                submition_df = pd.DataFrame(data=submition_data, index=[0])

                leaderboard = pd.concat([leaderboard, submition_df], axis=0, ignore_index=False)

                leaderboard = leaderboard.sort_values(by=['score', 'date'], ascending=[False, True]).reset_index(drop=True)

                leaderboard['ranking'] = leaderboard["score"].rank(ascending=False, method='first').astype(int)

                submition_data['date'] = submition_data.get('date').isoformat()
                submition_data['preds'] = str(dataframe['LABELS'].to_dict())

                st.write('Leaderboard actualizado')
                st.write(leaderboard[['ranking', 'user_name', 'file_name', 'date', 'score']])
                table.create(submition_data)


        else:
            st.write(f'Che el tamaño no está bien, es {dataframe.shape} cuando tiene que ser {shape_submit}')
            st.write(dataframe['LABELS'])

    except:
        st.write('Che el formato no está bien, tiene que ser solo una tira de 1 y 0 sin nombre')
