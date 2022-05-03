import numpy as np
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder, JsCode
import pandas as pd
from datetime import datetime
import streamlit as st


def render_grid(data):
    
    '''
    
    '''
    data = data.copy()
    data.rename(columns={'file_name':'Nombre archivo', 'date':'Fecha', 'score':'Score','user_name':'Datarocker',\
                        'ranking':'Ranking'},inplace=True)
    data = data[['Ranking','Datarocker','Score','Fecha','Nombre archivo']]
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(enabled=False)
    gb.configure_side_bar(
        filters_panel=False,
        columns_panel=False)
    
    gb.configure_default_column(
        groupable=True,
        value=True,
        filter='agTextColumnFilter',
        floatingFilter=True
        )
    gb.configure_columns(
        ['Datarocker','Fecha', 'Nombre archivo'],
        minWidth=300)
    gridOptions = gb.build()
    
    selected = AgGrid(
        data,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enableBrowserTooltips=True,
        tooltipShowDelay=0,
        theme='streamlit'#'dark'
    )
    return data


