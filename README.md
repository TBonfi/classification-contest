# Vamos los concursos de data science low cost

En un solo script `/concurso.py` tenemos una mini app que permite subir un file tipo txt/csv con las predicciones (0 o 1 en este caso) de una clasificación y obtener el F1 score.
Además guarda algunos datos referidos a la persona y el archivo subido para armar el ranking.

Queda pendiente armar algún dashboard y mejorar el UX porque es super rústico.

# Persistencia de datos

Se logra mediante el uso de Airtable y la librería :pyairtable: para leer la tabla leaderboard (donde incluimos el baseline) y la de y_true donde tenemos las predicciones enmascaradas.

# Aprendizajes:

1. Airtable tiene 1200 líneas de tope en el free tier y pensé que podíamos adjuntar el archivo (porque se puede) pero es necesario hostearlo y es un toque un dolor de cabeza. Resuelto mediante el guardado de las preds en un diccionario como texto porque tenemos hasta 100k de caracteres disponibles en las columnas tipo long text
2. pyairtable desordena las tablas, ojo si tenés dependencias, yo creé una columna que incremente y la usé como indice para y_true.
3. Hay que revisar y guardar los requirements para que Streamlit Cloud los levante.

# Backlog:

1. Gráficos lindos: datos de submition totales y de personas, heatmap tipo github de submitions, alguna animación cuando pasas tu benchmark anterior, algo para que escribas una vez tu nombre y dsp puedas buscarlo en vez de tipearlo.