import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer, StandardScaler

def exploracion_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col} tiene los siguientes valores únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()))    
    
def relacion_vr_categoricas(dataframe, variable_respuesta, paleta='bright', tamaño_graficas=(15, 10)):
    """
    Genera gráficos de barras para explorar la relación entre variables categóricas y una variable numérica de respuesta.

    Params:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    - variable_respuesta (str): El nombre de la columna numérica que se usará como variable respuesta.
    - paleta (str, opcional): Paleta de colores a utilizar en los gráficos. Por defecto es 'bright'.
    - tamaño_graficas (tuple, opcional): Tamaño de las gráficas en formato (ancho, alto). Por defecto es (15, 10).

    Returns:
    - None: La función muestra gráficos de barras y despliega los cinco primeros valores del DataFrame agrupado para cada variable categórica.
    """
    df_cat = separar_dataframe(dataframe)[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_graficas)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = (
            dataframe.groupby(columna)[variable_respuesta]
            .mean()
            .reset_index()
            .sort_values(variable_respuesta, ascending=False)
        )
        display(datos_agrupados.head())
        sns.barplot(
            x=columna,
            y=variable_respuesta,
            data=datos_agrupados,
            ax=axes[indice],
            palette=paleta,
        )
        axes[indice].tick_params(rotation=45)
        axes[indice].set_title(f'Relación entre {columna} y {variable_respuesta}')
        axes[indice].set_xlabel('')
    
    plt.tight_layout()
    plt.show()


def relacion_numericas(dataframe, variable_respuesta, paleta='bright', tamaño_graficas=(15, 10)):
    """
    Genera gráficos de dispersión para explorar la relación entre variables numéricas y una variable de respuesta.

    Params:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    - variable_respuesta (str): El nombre de la columna que se usará como variable respuesta.
    - paleta (str, opcional): Paleta de colores a utilizar en los gráficos. Por defecto es 'bright'.
    - tamaño_graficas (tuple, opcional): Tamaño de las gráficas en formato (ancho, alto). Por defecto es (15, 10).

    Returns:
    - None: La función muestra gráficos de dispersión para cada variable numérica en relación con la variable respuesta.
    """
    numericas = separar_dataframes(dataframe)[0]
    cols_numericas = numericas.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_graficas)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
            pass
        else:
            sns.scatterplot(
                x=columna,
                y=variable_respuesta,
                data=numericas,
                ax=axes[indice],
                palette=paleta,
            )
        axes[indice].set_title(f'Relación entre {columna} y {variable_respuesta}')
        axes[indice].set_xlabel(columna)
        axes[indice].set_ylabel(variable_respuesta)

    plt.tight_layout()
    plt.show()


def matriz_correlacion(dataframe, columnas=None, figsize=(8, 6), cmap='coolwarm', title='Matriz de Correlación'):
    """
    Genera y visualiza una matriz de correlación con opciones de personalización.

    Params:
    - dataframe (pd.DataFrame): DataFrame que contiene los datos.
    - columnas (list, optional): Lista de columnas específicas para calcular la correlación. Si no se especifica, usa todas las columnas numéricas.
    - figsize (tuple, optional): Tamaño de la figura (ancho, alto). Default es (8, 6).
    - cmap (str, optional): Mapa de colores para el heatmap. Default es 'coolwarm'.
    - title (str, optional): Título de la matriz de correlación. Default es 'Matriz de Correlación'.

    Returns:
    - None. Muestra el heatmap de correlación.
    """
    # Seleccionar columnas específicas si se proporcionan
    if columnas:
        matriz_corr = dataframe[columnas].corr()
    else:
        matriz_corr = dataframe.corr(numeric_only=True)

    # Crear máscara para la matriz triangular superior
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))

    # Configurar la figura
    plt.figure(figsize=figsize)
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        mask=mascara,
        cbar_kws={"shrink": 0.8}
    )

    # Título y ajustes
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()


def detectar_outliers(dataframe, color='red', tamaño_grafica=(15, 10)):
    """
    Genera gráficos de cajas (boxplots) para detectar outliers en las variables numéricas de un DataFrame.

    Params:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    - color (str, opcional): Color a utilizar en los boxplots. Por defecto es 'red'.
    - tamaño_grafica (tuple, opcional): Tamaño de las gráficas en formato (ancho, alto). Por defecto es (15, 10).

    Returns:
    - None: La función muestra gráficos de cajas para cada variable numérica.
    """
    df_num = separar_dataframes(dataframe)[0]
    num_filas = math.ceil(len(df_num.columns) / 2)
    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamaño_grafica)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(
            x=columna,
            data=df_num,
            ax=axes[indice],
            color=color,
        )
        axes[indice].set_title(f'Outliers de {columna}')
        axes[indice].set_xlabel('')

    plt.tight_layout()
    plt.show()


def plot_outliers_univariados(dataframe, columnas_numericas, tipo_grafica, bins, whis=1.5):
    fig, axes = plt.subplots(nrows=math.ceil(len(columnas_numericas) / 2), ncols=2, figsize= (15,10))

    axes = axes.flat

    for indice,columna in enumerate(columnas_numericas):

        if tipo_grafica.lower() == 'h':
            sns.histplot(x=columna, data=dataframe, ax= axes[indice], bins= bins)

        elif tipo_grafica.lower() == 'b':
            sns.boxplot(x=columna, 
                        data=dataframe, 
                        ax=axes[indice], 
                        whis=whis, #para bigotes
                        flierprops = {'markersize': 2, 'markerfacecolor': 'red'})
        else:
            print('No has elegido grafica correcta')
    
        axes[indice].set_title(f'Distribucion columna {columna}')
        axes[indice].set_xlabel('')

    if len(columnas_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()

def identificar_outliers_iqr(dataframe,columnas_numericas ,k =1.5):
    diccionario_outliers = {}
    for columna in columnas_numericas:
        Q1, Q3 = np.nanpercentile(dataframe[columna], (25,75)) #esta no da problemas con nulos
        iqr = Q3 -Q1

        limite_superior = Q3 + (iqr * k)
        limite_inferior = Q1 - (iqr * k)

        condicion_superior = dataframe[columna] > limite_superior
        condicion_inferior = dataframe[columna] < limite_inferior

        df_outliers = dataframe[condicion_superior | condicion_inferior]
        print(f'La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers')
        if not df_outliers.empty: #hacemos esta condicion por si acaso mi columna no tiene outliers
            diccionario_outliers[columna] = df_outliers

    return diccionario_outliers

def visualizar_categoricas(dataframe, lista_col_cat, variable_respuesta, bigote=1.5, paleta = 'bright',tipo_grafica='boxplot', tamaño_grafica=(15,10), metrica_barplot = 'mean',):
    num_filas = math.ceil(len(lista_col_cat)/ 2)

    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_grafica)

    axes = axes.flat

    for indice, columna in enumerate(lista_col_cat):
        if tipo_grafica.lower()=='boxplot':
            sns.boxplot(x=columna, 
                        y=variable_respuesta, 
                        data=dataframe,
                        whis=bigote,
                        hue=columna,
                        legend=False,
                        ax= axes[indice])
            
        elif tipo_grafica.lower()== 'barplot':
            sns.barplot(x=columna,
                        y=variable_respuesta,
                        ax = axes[indice],
                        data=dataframe,
                        estimator=metrica_barplot,
                        palette= paleta)
        else:
            print('No has elegido una grafica correcta')

        axes[indice].set_title(f'Relacion {columna} con {variable_respuesta}')
        axes[indice].set_xlabel('')
        plt.tight_layout()

def separar_dataframes(dataframe):
    """
    Separa un DataFrame en dos DataFrames: uno con variables numéricas y otro con variables categóricas.

    Params:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos a analizar.

    Returns:
    - tuple: 
        - pd.DataFrame: DataFrame con las columnas numéricas.
        - pd.DataFrame: DataFrame con las columnas categóricas (incluye objetos y categorías).
    """
    return dataframe.select_dtypes(include=np.number), dataframe.select_dtypes(include=['O', 'category'])

def plot_numericas(dataframe, figsize=(10, 8)):
    """
    Genera histogramas para todas las columnas numéricas de un DataFrame, incluyendo etiquetas en los valores.

    Params:
    - dataframe (pd.DataFrame): El DataFrame que contiene las columnas numéricas a graficar.
    - figsize (tuple, opcional): Tamaño de la figura en formato (ancho, alto). Por defecto es (10, 8).

    Returns:
    - None: La función genera y muestra histogramas para cada columna numérica, con etiquetas en las barras.
    """
    cols_numericas = dataframe.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        ax = axes[indice]
        plot = sns.histplot(
            x=columna,
            data=dataframe,
            ax=ax,
            bins=50
        )
        ax.set_title(columna)
        ax.set_xlabel('')
        
        # Agregar etiquetas en los valores
        for patch in plot.patches:
            height = patch.get_height()
            if height > 0:  # Solo etiqueta si el valor es mayor a 0
                ax.annotate(
                    f'{int(height)}',
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def plot_categoricas(dataframe, paleta="bright", tamano_grafica=(15, 8)):
    """
    Grafica la distribución de las variables categóricas del DataFrame.

    Parameters:
    - dataframe (DataFrame): El DataFrame con las variables categóricas.
    - paleta (str, opcional): La paleta de colores a utilizar en las gráficas. Por defecto es "bright".
    - tamano_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 8).
    """
    # Calcular la disposición de los subgráficos
    num_columnas = len(dataframe.columns)
    filas = 2
    columnas = math.ceil(num_columnas / filas)
    
    fig, axes = plt.subplots(filas, columnas, figsize=tamano_grafica)
    axes = axes.flatten()

    for indice, columna in enumerate(dataframe.columns):
        sns.countplot(
            x=columna, 
            data=dataframe, 
            order=dataframe[columna].value_counts().index,
            ax=axes[indice], 
            palette=paleta
        )
        axes[indice].tick_params(axis='x', rotation=90)  # Rotar etiquetas del eje X
        axes[indice].set_title(columna, fontsize=12)
        axes[indice].set(xlabel=None, ylabel=None)

    # Eliminar subgráficos vacíos
    for i in range(num_columnas, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

def relacion_vr_numericas_problema_categorico(df, vr):
    df_num, df_cat = separar_dataframes(df)
    columnas_numericas = df_num.columns
    num_filas = math.ceil(len(columnas_numericas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=(15,10))
    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):
        sns.histplot(df, x=columna, ax=axes[indice], hue=vr , bins=20)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(columnas_numericas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()

def detectar_orden_cat(df, lista_cat, var_respuesta):
    lista_ordenadas = []
    lista_desordenadas = []
    for categorica in lista_cat:
        print(f'Estamos evaluando la variable {categorica.upper()}')
        df_cross_tab = pd.crosstab(df[categorica], df[var_respuesta])
        display(df_cross_tab)

        chi2, p, dof, excepted = chi2_contingency(df_cross_tab) 
        
        if p < 0.05:
            print(f'Sí tiene orden la variable {categorica}')
            lista_ordenadas.append(categorica)
        else:
            print(f'La variable {categorica} no tiene orden')
            lista_desordenadas.append(categorica)

    return lista_ordenadas, lista_desordenadas

def plot_relacion(self, vr, tamano_grafica=(40, 12)):


        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma"
                              )
                
def graficar_relaciones_categoricas(dataframe, vr):
    """
    Genera gráficos de barras en un subplot para todas las columnas categóricas
    mostrando su relación con la variable de respuesta.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame que contiene los datos.
    - vr (str): Nombre de la variable de respuesta.
    """
    # Seleccionar columnas categóricas del DataFrame
    lista_cat = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Configurar el número de filas y columnas del subplot
    num_columnas = len(lista_cat)
    cols = 2  # Número de gráficos por fila
    filas = math.ceil(num_columnas / cols)  # Calcula las filas necesarias
    
    # Configurar la figura
    fig, axes = plt.subplots(nrows=filas, ncols=cols, figsize=(16, 6 * filas))
    axes = axes.flatten()  # Asegura que axes sea una lista para iterar fácilmente
    
    # Iterar sobre las columnas categóricas y crear gráficos
    for indice, columna in enumerate(lista_cat):
        sns.countplot(x=columna, hue=vr, data=dataframe, ax=axes[indice], palette="magma")
        axes[indice].set_title(f"Relación {columna} vs {vr}")
        axes[indice].set_xlabel(columna)
        axes[indice].set_ylabel("Frecuencia")
        axes[indice].legend(title=f"{vr}")
        axes[indice].tick_params(axis='x', rotation=45)
    
    # Ocultar cualquier gráfico sobrante si hay menos categorías que subplots
    for j in range(indice + 1, len(axes)):
        fig.delaxes(axes[j])  # Elimina subplots vacíos
    
    # Ajustar la disposición
    plt.tight_layout()
    plt.show()

def graficar_relaciones_numericas(dataframe, vr):
    """
    Genera gráficos de histogramas en un subplot para todas las columnas numéricas
    mostrando su relación con la variable de respuesta, omitiendo la variable de respuesta misma.
    
    Args:
    - dataframe (pd.DataFrame): DataFrame que contiene los datos.
    - vr (str): Nombre de la variable de respuesta.
    """
    # Seleccionar columnas numéricas del DataFrame
    lista_num = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Configurar el número de filas y columnas del subplot
    num_columnas = len(lista_num)
    cols = 2  # Número de gráficos por fila
    filas = math.ceil(num_columnas / cols)  # Calcula las filas necesarias
    
    # Configurar la figura
    fig, axes = plt.subplots(nrows=filas, ncols=cols, figsize=(16, 6 * filas))
    axes = axes.flatten()  # Asegura que axes sea una lista para iterar fácilmente
    
    # Contador para manejar los subplots correctamente
    plot_index = 0
    
    # Iterar sobre las columnas numéricas y crear gráficos
    for columna in lista_num:  # Itera solo sobre las columnas numéricas
        if columna == vr:  # Si la columna es la variable de respuesta, omitirla
            continue
        
        sns.histplot(x=columna, 
                     hue=vr, 
                     data=dataframe, 
                     ax=axes[plot_index], 
                     palette="magma", 
                     legend=True)  # Activa la leyenda
        axes[plot_index].set_title(f"Relación {columna} vs {vr}")
        axes[plot_index].set_xlabel(columna)
        axes[plot_index].set_ylabel("Frecuencia")
        axes[plot_index].tick_params(axis='x', rotation=45)
        plot_index += 1  # Incrementar el índice del subplot
    
    # Ocultar cualquier gráfico sobrante si hay menos variables que subplots
    for j in range(plot_index, len(axes)):
        fig.delaxes(axes[j])  # Elimina subplots vacíos
    
    # Ajustar la disposición
    plt.tight_layout()
    plt.show()

def visualizar_escalados_completos(df, lista_num):
    """
    Genera subplots para boxplots e histogramas de las columnas numéricas originales
    y sus versiones escaladas, cubriendo todas las columnas.
    
    Args:
    - df (pd.DataFrame): DataFrame con las columnas originales y escaladas.
    - lista_num (list): Lista de nombres de columnas numéricas originales.
    """
    # Sufijos para las versiones escaladas
    sufijos = ["robust", "minmax", "norm", "stand", ""]
    
    # Crear lista completa de columnas originales y escaladas
    columnas_totales = [f"{col}_{sufijo}" if sufijo else col for col in lista_num for sufijo in sufijos]
    
    # Número de columnas por fila en los subplots
    num_columnas = 5  # Máximo 5 gráficos por fila
    num_filas = math.ceil(len(columnas_totales) / num_columnas)  # Calcula filas necesarias
    
    # Crear figura
    fig, axes = plt.subplots(nrows=num_filas, ncols=num_columnas, figsize=(20, 5 * num_filas))
    axes = axes.flatten()  # Asegura que los ejes sean iterables
    
    # Iterar sobre las columnas para graficar
    for indice, col in enumerate(columnas_totales):
        if col in df.columns:  # Verifica que la columna exista en el DataFrame
            # Alternar entre boxplots e histogramas
            if "robust" in col or "minmax" in col or "norm" in col or "stand" in col:
                sns.boxplot(x=col, data=df, ax=axes[indice])
            else:
                sns.histplot(x=col, data=df, ax=axes[indice], bins=50)
            axes[indice].set_title(f"{col}")
        else:
            axes[indice].set_visible(False)  # Oculta subplots sin datos
    
    # Ocultar cualquier gráfico sobrante
    for j in range(len(columnas_totales), len(axes)):
        fig.delaxes(axes[j])
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()


def escalar_columnas(df, columnas_numericas):
    """
    Escala las columnas numéricas del DataFrame utilizando diferentes técnicas
    y agrega las columnas escaladas al DataFrame original.
    
    Args:
    - df (pd.DataFrame): DataFrame original.
    - columnas_numericas (list): Lista de nombres de columnas numéricas a escalar.
    
    Returns:
    - pd.DataFrame: DataFrame con las columnas escaladas añadidas.
    """
    # Inicializar escaladores
    escaladores = {
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "norm": Normalizer(),
        "stand": StandardScaler()
    }
    
    for nombre, escalador in escaladores.items():
        # Aplicar el escalador a las columnas numéricas
        datos_transformados = escalador.fit_transform(df[columnas_numericas])
        
        # Crear nombres para las columnas escaladas
        nombres_columnas = [f"{col}_{nombre}" for col in columnas_numericas]
        
        # Agregar las columnas transformadas al DataFrame original
        df[nombres_columnas] = datos_transformados
    
    return df

def escalar_columnas_metodo(df, columnas_numericas, metodo_escalador):
    """
    Escala directamente las columnas numéricas del DataFrame utilizando un método específico
    y guarda el escalador utilizado en un archivo pickle.
    
    Args:
    - df (pd.DataFrame): DataFrame original.
    - columnas_numericas (list): Lista de nombres de columnas numéricas a escalar.
    - metodo_escalador (str): Método de escalado a utilizar ('robust', 'minmax', 'norm', 'stand').
    - archivo_pickle (str): Nombre del archivo donde se guardará el escalador.
    
    Returns:
    - pd.DataFrame: DataFrame con las columnas numéricas escaladas.
    """
    # Inicializar los escaladores disponibles
    escaladores = {
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "norm": Normalizer(),
        "stand": StandardScaler()
    }
    
    # Verificar si el método de escalado es válido
    if metodo_escalador not in escaladores:
        raise ValueError(f"El método '{metodo_escalador}' no es válido. Usa uno de: {list(escaladores.keys())}")
    
    # Seleccionar el escalador
    scaler = escaladores[metodo_escalador]
    
    # Aplicar el escalado directamente a las columnas originales
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
    return df


def plot_top_numericas(dataframe, columna_categoria, top_n=10, figsize=(10, 8)):
    """
    Plotea gráficos de barras para las columnas numéricas del DataFrame,
    mostrando el top_n valores más altos de cada columna numérica.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame con los datos.
    - columna_categoria (str): Nombre de la columna categórica a usar en el eje Y (ejemplo: Provincia).
    - top_n (int): Número de filas principales a incluir según cada columna numérica.
    - figsize (tuple): Tamaño de la figura de los gráficos de barras.
    """
    # Seleccionar solo las columnas numéricas
    cols_numericas = dataframe.select_dtypes(include='number').columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    
    # Crear subgráficos
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        # Ordenar el DataFrame por la columna actual y tomar el top_n
        top_dataframe = dataframe.sort_values(by=columna, ascending=False).head(top_n)

        # Crear un gráfico de barras con columna_categoria en el eje Y
        sns.barplot(y=columna_categoria, x=columna, data=top_dataframe, ax=axes[indice], palette='viridis')
        axes[indice].set_title(f'Top {top_n} - {columna}')
        axes[indice].set_xlabel('')
        axes[indice].set_ylabel(columna_categoria)
    
    # Eliminar gráficos vacíos si el número de columnas numéricas es impar
    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()

def crear_tabla_pivote(dataframe, index, columns, values, aggfunc='count'):
    """
    Crea una tabla pivote a partir de un DataFrame.

    Params:
    - dataframe (pd.DataFrame): El DataFrame de entrada.
    - index (str): Columna que se usará como índice (filas) en la tabla pivote.
    - columns (str): Columna que se usará como encabezado de columnas en la tabla pivote.
    - values (str): Columna cuyos valores se resumirán en la tabla pivote.
    - aggfunc (str, opcional): Función de agregación para resumir los datos. Por defecto es 'count'.

    Returns:
    - pd.DataFrame: La tabla pivote generada.
    """
    return dataframe.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc
    )

# ----- Funciones consulta ----- #

def consulta_tipo_accidente(df, anio, filtro_columna, filtro_valor):
    """
    Consulta los tipos de accidente por año en base a región o provincia.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        anio (int): Año para filtrar los datos.
        filtro_columna (str): La columna por la cual filtrar (e.g., 'region' o 'provincia').
        filtro_valor (str): El valor específico de la columna por el cual filtrar.

    Returns:
        pd.DataFrame: DataFrame con el conteo de tipos de accidente para el año y filtro especificado.
    """
    try:
        # Filtrar por año
        df_filtrado = df[df['anio'] == anio]

        # Filtrar por columna (región o provincia)
        df_filtrado = df_filtrado[df_filtrado[filtro_columna] == filtro_valor]

        # Contar los tipos de accidente
        resultado = df_filtrado['tipo_accidente'].value_counts().reset_index()
        resultado.columns = ['Tipo de Accidente', 'Cantidad']

        return resultado

    except KeyError as e:
        return f"Error: La columna '{filtro_columna}' o 'tipo_accidente' no existe en el DataFrame."
    except Exception as e:
        return f"Hubo un error: {str(e)}"
    

def comparar_tipo_accidente(df, anio, filtro_columna, valor1, valor2):
    """
    Compara los tipos de accidente entre dos regiones o provincias para un año específico.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        anio (int): Año para filtrar los datos.
        filtro_columna (str): La columna por la cual filtrar (e.g., 'region' o 'provincia').
        valor1 (str): Primer valor de la columna para comparar.
        valor2 (str): Segundo valor de la columna para comparar.

    Returns:
        pd.DataFrame: DataFrame con el conteo de tipos de accidente para cada región o provincia.
    """
    try:
        # Filtrar por año
        df_filtrado = df[df['anio'] == anio]

        # Filtrar por el primer valor
        df_valor1 = df_filtrado[df_filtrado[filtro_columna] == valor1]
        conteo_valor1 = df_valor1['tipo_accidente'].value_counts().reset_index()
        conteo_valor1.columns = ['Tipo de Accidente', f'Cantidad en {valor1}']

        # Filtrar por el segundo valor
        df_valor2 = df_filtrado[df_filtrado[filtro_columna] == valor2]
        conteo_valor2 = df_valor2['tipo_accidente'].value_counts().reset_index()
        conteo_valor2.columns = ['Tipo de Accidente', f'Cantidad en {valor2}']

        # Combinar los resultados en un solo DataFrame
        resultado = pd.merge(conteo_valor1, conteo_valor2, on='Tipo de Accidente', how='outer').fillna(0)

        return resultado

    except KeyError as e:
        return f"Error: La columna '{filtro_columna}' o 'tipo_accidente' no existe en el DataFrame."
    except Exception as e:
        return f"Hubo un error: {str(e)}"
    
def consulta_accidentes_interactiva(data):
    """
    Consulta interactiva de los tipos de accidente por región o provincia, solicitando datos al usuario.

    Args:
        data (pd.DataFrame): DataFrame con los datos de accidentes.

    Returns:
        pd.DataFrame: DataFrame con la cantidad de accidentes agrupados por tipo, ordenados de mayor a menor.
    """
    # Solicitar inputs al usuario
    region = input("Ingresa la región, ejemplo: 'Región Ozama'(o presiona Enter para omitir): ").strip()
    provincia = input("Ingresa la provincia, ejemplo: 'Santo Domingo (o presiona Enter para omitir): ").strip()
    anio = input("Ingresa el año entre 2005 - 2023(o presiona Enter para omitir): ").strip()

    # Filtrado inicial
    filtro = data.copy()
    if region:
        filtro = filtro[filtro['region'] == region]
    if provincia:
        filtro = filtro[filtro['provincia'] == provincia]
    if anio:  # Convertir año a número si se proporciona
        anio = int(anio)
        filtro = filtro[filtro['anio'] == anio]

    # Agrupar por tipo de accidente y contar
    resultado = filtro.groupby('tipo_accidente').size().reset_index(name='cantidad')
    resultado = resultado.sort_values(by='cantidad', ascending=False)  # Orden descendente
    return resultado

def comparar_accidentes_interactiva(data):
    """
    Compara interactivamente los tipos de accidente entre dos regiones o provincias, solicitando datos al usuario.

    Args:
        data (pd.DataFrame): DataFrame con los datos de accidentes.

    Returns:
        pd.DataFrame: DataFrame con la comparación de accidentes por tipo entre las regiones o provincias, ordenados de mayor a menor.
    """
    # Solicitar inputs al usuario
    region1 = input("Ingresa la primera región (o presiona Enter para omitir): ").strip()
    provincia1 = input("Ingresa la primera provincia (o presiona Enter para omitir): ").strip()
    region2 = input("Ingresa la segunda región (o presiona Enter para omitir): ").strip()
    provincia2 = input("Ingresa la segunda provincia (o presiona Enter para omitir): ").strip()
    anio = input("Ingresa el año (o presiona Enter para omitir): ").strip()

    # Validar inputs
    if not region1 and not provincia1:
        raise ValueError("Debes proporcionar al menos una región o provincia para la primera comparación.")
    if not region2 and not provincia2:
        raise ValueError("Debes proporcionar al menos una región o provincia para la segunda comparación.")

    # Filtrar datos según los inputs
    if region1:
        filtro1 = data[data['region'] == region1]
    else:
        filtro1 = data[data['provincia'] == provincia1]

    if region2:
        filtro2 = data[data['region'] == region2]
    else:
        filtro2 = data[data['provincia'] == provincia2]

    if anio:  # Filtrar por año si se proporciona
        anio = int(anio)
        filtro1 = filtro1[filtro1['anio'] == anio]
        filtro2 = filtro2[filtro2['anio'] == anio]

    # Agrupar por tipo de accidente y contar
    resultado1 = filtro1.groupby('tipo_accidente').size().reset_index(name=f'{region1 or provincia1}')
    resultado2 = filtro2.groupby('tipo_accidente').size().reset_index(name=f'{region2 or provincia2}')

    # Combinar los resultados en un único DataFrame para comparación
    comparacion = pd.merge(resultado1, resultado2, on='tipo_accidente', how='outer').fillna(0)

    # Convertir las columnas de conteo a enteros
    comparacion.iloc[:, 1:] = comparacion.iloc[:, 1:].astype(int)

    # Ordenar el DataFrame combinado en orden descendente por la suma de las columnas de accidentes
    comparacion['total'] = comparacion.iloc[:, 1:].sum(axis=1)  # Crear una columna temporal de suma total
    comparacion = comparacion.sort_values(by='total', ascending=False).drop(columns=['total'])  # Orden descendente y eliminar columna temporal

    return comparacion

def reemplazar_valores(dataframe, columna, mapeo):
    """
    Reemplaza valores en una columna específica de un DataFrame utilizando un diccionario de mapeo.

    Params:
    - dataframe (DataFrame): El DataFrame donde se aplicarán los cambios.
    - columna (str): El nombre de la columna en la que se reemplazarán los valores.
    - mapeo (dict): Un diccionario donde las claves son los valores a reemplazar
                    y los valores son los nuevos valores.

    Returns:
    - DataFrame: El DataFrame con los valores reemplazados en la columna especificada.
    """
    dataframe[columna] = dataframe[columna].replace(mapeo)
    return dataframe
