#!/usr/bin/env python
# coding: utf-8

# # PRUEBA DATA SCIENTIST GLOBANT

# ##### Julián Cujabante V.
# - Entrega antes del Martes 10 de Noviemnre del 2020.

# ### CARGA DE BASE DE DATOS

# In[431]:


import pandas as pd
import numpy as np
import pandas_profiling


# In[432]:


data=pd.read_csv(r'E:\Users\julian\Globant\dataset_diabetes\diabetic_data.csv', sep=',',engine='python')


# In[433]:


data.head()


# - Las dimensiones del data frame son más de 100k observaciones y 50 variables, entre las cuales se incluye la variable dependiente.

# In[434]:


data.shape


# - La distribución de frecuencia de la variable dependiente:

# In[435]:


print(data['readmitted'].value_counts())


# - Cómo puede observarse en los datos cuando hay un missing se codifica como ?, se codificará como missing en Python.

# In[436]:


data=data.replace(['?'],np.nan)


# In[437]:


data.head()


# - Se revisa la cantidad de missings presentes en la base de datos

# In[438]:


data.isnull().sum()


# - Note en cuales variables se ubican los missing: 'race', 'weigth', 'payer code', 'medical specialty', y variables de dianostico. Todas son variables categóricas. Existen varias aproximaciones para tratar con los missing, entr eellas, eliminar las observaciones (aproximación que no se llevará a cabo, pues la cantidad de missing es muy alta. Puede de igual manera, imputarse los valores de estos missings, ya sea con la media de la variable, o con métodos como KNN. No obstante, en algunas de las variables, imputar los valores tendría implicaciones fuertes, cómo lo es en el caso de diagnóstico, imputar a una persona con la media de ocurrencia un diagnostico que no corresponde, podría afectar la predicción de la variable dependiente. Se opta por otra aproximación que coincide con la llevada a cabo en el paper asociado a las base de datos: eliminar algunas de estas variables, como por ejemplo: 'payer_code', 'weight'.

# - Revisando la estructura de la base de datos, se observa el tipo de datos bajo el cúal están codificadas algunas variables:

# In[439]:


data.info()


# - Observe que alguans de las variables están codificadas en un tipo de dato que posiblemnete no le corresponda. Ejemplo de esto son los identificadores de encuentro y de paciente, que están codificados como números, cuando en realidad no tiene sentido que lo sean. De ser números, piense si tendría sentido realizar una operación aritmética entre estos. De igual manera, algunas de las variables que se consiera capturan clases o tipos están codificadas como objectos (string), se cambiarán a la clase catagoria.

# In[440]:


data.loc[:, data.dtypes == 'object'] =data.select_dtypes(['object']).apply(lambda x: x.astype('category'))


# In[441]:


data.info()


# - Cuantas diferentes clases hay en las variables categóricas?

# In[442]:


print("Número de categorias en: ")
for ColName in data.select_dtypes(['category']):
    print("{} = {}".format(ColName,       len(data[ColName].unique())))


# - Se observa como hay unas variables categóricas que tienen muchas categorías, cómo lo son los códigos de diagnóstico basados en la codificación ICD9. Asimismo, se nota una gran cantidad de categorías en la variable: 'medical specialty' de quién remitió.

# - Se cambia la codificación de tipo numérico que tienen las variables identificadoras (id's).

# In[443]:


data=data.astype({'encounter_id':'category','patient_nbr':'category','admission_type_id': 'category','discharge_disposition_id': 'category','admission_source_id': 'category'})


# - Se considera que un manejo de las categorías de diagnóstico al agruparlas según su clasificación ICD9 podrí areducir la cantidad de categorías y permitir una clasificación más efectiva pro parte de la máquina de aprendizaje empleada. Se realizará en ese orden de ideas, un procedimiento similar al realizado por los autores en el Paper relacionado con esta base de datos.

# - Se agrupan las categorias de diagnostico 1 en los grupos codificados en ICD9

# In[444]:


def cat(x):
    if x>='1' and x<='139': 
        return "Infectious and parasitic diseases"
    if x>='140' and x<='239':
        return "Neoplasms"
    if x>='240' and x<='279':
        return 'Endocrine nutritional and metabolic diseases'
    if x>='280' and x<='289':
        return 'Diseases of the blood and blood forming organs'
    if x>='290' and x<='319':
        return 'Mental disorders'
    if x>='320' and x<='389':
        return 'Diseases of the nervous system and sense organs'
    if x>='390' and x<='459':
        return 'Diseases of the circulatory system'
    if x>='460' and x<='519':
        return 'Diseases of the respiratory system'
    if x>='520' and x<='579':
        return 'Diseases of the digestive system'
    if x>='580' and x<='629':
        return 'Diseases of the genitourinary system'
    if x>='630' and x<='679':
        return 'Complications of pregnancy childbirth and the puerperrium'
    if x>='680' and x<='709':
        return 'Diseases of the skin and subcutaneous tissue'
    if x>='710' and x<='739':
        return 'Diseases of the musculoskeletal system and cognitive'
    if x>='740' and x<='759':
        return 'Congetinal anomalies'
    if x>='760' and x<='779':
        return 'Certain conditions originating in the perinatal period'
    if x>='780' and x<='799':
        return 'Symptoms signs and ill defined conditions'
    if x>='800' and x<='999':
        return 'Injury and poisoning'
    if x>='V01' and x<='V89':
        return 'Suplementary classification of factors influencing health status and contact with health services'
    if x>='E800' and x<='E900':
        return 'Suplementary classification of external causes of injury and poisoning'
    if x=='8':
        return 'Infectious and parasitic diseases'


# In[445]:


data['diagnose_1'] = data['diag_1'].apply(lambda x: cat(x))


# - A pesar que se redujeron categorías siguen siendo una cantidad considerable.
# - Se cuenta la distribución de estas categorias en la variable diagnose_1

# In[446]:


pd.value_counts(data['diagnose_1'])


# - Se puede ver cómo hay algunas de las categorías que se repiten mucho y otras que no tienen casi frecuencia.
# - Se ve esta misma información en términos porcentuales

# In[447]:


100 * data['diagnose_1'].value_counts() / len(data['diagnose_1'])


# - Se gráfica la distribución de categorías en la variable recién creada: 'diagnose_1'.

# In[448]:


import matplotlib.pyplot as plt
import seaborn as sns
diagnose1_count = data['diagnose_1'].value_counts()
sns.set(style="darkgrid")
sns.barplot(diagnose1_count.values,diagnose1_count.index, alpha=0.9)
plt.title('Distribución frecuencia de diagnose_1')
plt.ylabel('diagnose_1', fontsize=13)
plt.xlabel('Número de diagnósticos', fontsize=14)
plt.show()


# - Al igual que como se hizo en el paper, se reducen las categorias al tomar las que más freuencia tengan y agrupar las demás en una categoría llamada 'other'. Se eligen las 5 categorías con más frecuencia.

# In[449]:


top_diagnose_1 = data['diagnose_1'].isin(data['diagnose_1'].value_counts().index[:5])
data.loc[~top_diagnose_1, 'diagnose_1'] = "other"


# - Se grafica de nuevo la variable categórica 'diagnose_1' luego de reducir la cantidad de categorias.

# In[450]:


diagnose1_count = data['diagnose_1'].value_counts()
sns.barplot(diagnose1_count.values,diagnose1_count.index, alpha=0.9)
plt.title('Distribución frecuencia de diagnose_1')
plt.ylabel('diagnose_1', fontsize=13)
plt.xlabel('Número de diagnósticos', fontsize=14)
plt.show()


# In[451]:


data=data.astype({'diagnose_1':'category'})


# - Se encuentra que hay más de 30k registros de números de pacientes que son duplicados.
# - Al igual que cómo se hizo en el paper, se eliminarán los repetidos y sólo se dejará el primer encuentro, la razón de esto, es que posiblemente un registro duplicado está conectado y correlacionado con los registros anteriores, piense en una enfermedad crónica. En ese orden de ideas estos casos podrían sesgar al algorito.

# In[452]:


data['patient_nbr'].duplicated().sum()


# In[453]:


data.drop_duplicates('patient_nbr',keep='first',inplace=True)


# - Queda esta cantidad de observaciones, luego de eliminar los registros duplicados de pacientes.

# In[454]:


len(data)


# - Hay algunas variables que se cree no son relevantes para el análisis, pues no se considera haya una relación clara entre estas y la variable a predecir. Estas por ejemplo son: 'payer code' (la manera con la que pago probablemente no se relacione con si tiene que volver a ser admitido o no)

# - Se eliminan variables

# In[455]:


data=data.drop(['payer_code','diag_1','diag_2','diag_3','patient_nbr','encounter_id','weight','race','medical_specialty','admission_type_id','discharge_disposition_id','admission_source_id'],axis=1)


# In[456]:


data.isnull().sum()


# - Esta es la dimensión del nuevo data frame: más de 70k observaciones y 39 variables.

# In[457]:


data.shape


# - Se separa la base de datos en su parte numércia y categórica.

# In[458]:


data_num = data.select_dtypes(include=['int64']).copy()


# In[459]:


data_cat = data.select_dtypes(include=['category']).copy()


# - La distribución de la variable dependiente luego de la eliminación de duplicados es:

# In[460]:


print(data['readmitted'].value_counts())


# - Importante recalcar que las proporciones de la variable a predecir disminuyeron considerablemente. Se puede ver que hay un posible desbalance en los datos frente a la categoria: '<30'.

# In[461]:


sns.countplot(data['readmitted'])


# ## EXPLORACIÓN DE LOS DATOS

# ### Datos numéricos

# - Se presentan estadísticas descriptivas de las variables numéricas

# In[462]:


data_num.describe()


# - Se soportará este análisis con las gráficas de distribución sobre estas variables.

# In[463]:


data.hist(figsize=(10,8),grid=False)


# - Note que la variable 'numb_lab_procedures' tiene cierta forma 'normal, con una gran acumulación de frecuencia cerca de su media.
# - La variable 'num_medications' tiende a concetrar más peso de su distribución en su cola izquierda. Los valores 'pequeños' son más frecuentes.
# - La variable 'num_procedures' aparece en saltos.
# - La variable 'num_diagnoses' esta sesgada hacia la derecha, al parecer son más frecuentes los valores más altos.
# - Las siguientes variables: 'number_emergency', 'number_inpatient', 'number_outpatient' concentran su mayoría de distribución en los valores más pequeños. Note que los valores máximos se alejan mucho de la media, probablemnete hayan outliers aqui.
# - Finalmente, la variable 'time_in_hospital' muestra que es más frecuente estar entre 1 y 5 días en el hospital, siendo menos fecuente las estancias largas.

# In[464]:


data['number_emergency'].hist(figsize=(10,10), grid=False)


# - Se realiza una matriz de correlación entre las variables:

# - Existe correlación positiva entre el numero de procedimientos, de pruebas de laboratorio y los días transcurridos en el hospital. Esto podría ser lógico, pues si un paciente está grave o al menos no estable como para ser remitido a su casa, indica que muy probablemente se le realicen varios procedimientos y pruebas médicas con el ánimo de curarlo. Es posible que haya cierta multicolinealidad entre estas variables.

# In[465]:


corr=data.corr()
ax=sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right');


# - Se grafican boxplots para ayudar a discernir sobre la distribución de estas variables.

# In[466]:


import matplotlib.pyplot as plt


# In[467]:


data_num.boxplot(figsize=(15,15))
    


# - Hay outliers, sobre todo en las variables: 'number_outpatient', 'number_emergency', 'number_inpatient', 'num_medications'. Estos valores pueden asociarse quizá con aquellas observaciones de pacientes 'graves'. Una aproximación para tratar los outliers es eliminarlos, pero se desiste de esta opción, pues no se quiere perder más datos, otra podría ser imputar los valores, pero como ya se vió en muchas de estas su valor se concentra en 1, de imputarseles este valor se perderia variación en estas variables, pues todas tendrían valor 1.

# In[468]:


data['number_outpatient'].hist()


# ##### Distribución de las variables categóricas

# In[582]:


fig, ax = plt.subplots(15,2, figsize=(80, 120))
for variable, subplot in zip(data_cat, ax.flatten()):
    sns.countplot(data_cat[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# ## FEATURE ENGINEERING

# - Se realizarán una serie de tratamientos y preprocesamientos para las variables de la base antes de ser pasadas por la máquina de aprendizaje.

# - Se obtiene el valor de la variable dependiente y se guarda.

# In[471]:


y=data.readmitted.values


# In[472]:


#data=data.drop(['readmitted'],axis=1)


# - Esta variable contiene muchas variables categóricas. Estás deben de procesarse antes de ser pasadas a los algortimos. La mayoria de los algortimos evaluados sólo permiten valores numéricos, por tal motivo, estas categorías deben de numerizarse. Una manera para hacerlo es con One Hot Encoding, o crear tantas variables binarias por cada categoría contenida en la variable. La razón paar hacer esto es que de numerizarse las categorias dentro de la misma variable, se le podría estar diciendo al algortimo que existe una especie de orden, cuando en realidad no es así. Piense en el ejemplo de 'gender' codificar las categorias de esta variable con númros en una sola variable daría que para mujer se usaria un úmero: digamos 1 y para hombres 2. Esto implicaría orden? o que la categoría hombre es dos veces la categoría mujer? no. Por tal motivo se usa el One Hot Encoding. Debe tenerse cuidado de eliminar unas de las variables binarias recién creadas para las categorías contenidas en una variable, para no caer en la trampa de la variable dicotoma, que es lo mismo que tener multicolinealidad perfecta si se incluyen todas las variables binarias de las categorias contenidas en una variable. Otro posible problema de esta aproximación es que se crean muchas variables y puede generar mucha complejidad en el modelo añadiendole varianza y en el peor de lso casos cayendo en la maldición de la dimensionalidad.

# ### One Hot Encoding

# - Teniendo en cuenta lo anterior se realizar la binariazción de las variables categóricas y se elimina una de las recién creadas variables binarias.

# In[473]:


data_dummies=pd.get_dummies(data.loc[:, data.columns != 'readmitted'],drop_first=True)


# - La dimensión del nuevo data frame es: las mismas observaciones, pero 80 variables.

# In[474]:


data_dummies.shape


# ### Frequency Encoding

# - Se realiza también un encoding de las variables categóricas basado en la frecuencia relativa de la categoría en cada varaible, esto evitaría generar muchas variables adicionales y se obtendría un valor numérico dela categoría. En ese orden de ideas, si la variable 'gender': tiene las categorías 'hombre ' y 'mujer', se calculará la frecuencia relativa de cada categoría dentro de la variable y se le asignará este valor numérico, pro ejemplo si la categoría mujer es un 76% en la variable 'gender', se le asignara esta frecuencia a cada observación que tenga esta categoría asignada.

# In[475]:


data_cat_freq = data_cat.loc[:, data_cat.columns != 'readmitted'].copy()
for c in data_cat_freq.columns.to_list():
    data_cat_freq[c] = data_cat_freq.groupby(c).transform('count')/len(data_cat_freq[c])


# In[476]:


frames_freq = [data_cat_freq, data_num]
data_freq = pd.concat(frames_freq, axis = 1)


# - Se mantiene el mismo número de variables.

# In[477]:


data_freq.shape


# - Se observa ahora las distribuciones de las varaibles.

# In[478]:


data_freq.info()


# In[479]:


data_freq.hist(figsize=(20,8), grid=False)


# In[480]:


data_freq.describe()


# In[481]:


corr_freq=data_freq.corr()
ax=sns.heatmap(corr_freq, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20,220,n=200),square=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right');


# ### Escalación de los datos 

# - Se realiza también la esclación de los datos para tener la misma escala o dimensión de las variables y permitir un mejor desempñeo en los algortimos de clasificación.

# In[482]:


col_namesd = data_dummies.columns.tolist()


# In[483]:


col_namesf = data_freq.columns.tolist()


# In[484]:


type(col_namesd)


# ##### MinMax

# - La esclación MinMax deja todas las variables con valores entre 0 y 1. Esta es útil en escenarios con muchas variables dummies, pues las deja cómo estan, y todas las demás variables en valores similares a estas dummies.

# In[485]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
scalerMM = MinMaxScaler()


# In[486]:


data_MMd = scalerMM.fit_transform(data_dummies)
data_scaledMMd=pd.DataFrame(data_MMd)
data_scaledMMd.columns=col_namesd
data_scaledMMd.head()


# In[487]:


data_MMf = scalerMM.fit_transform(data_freq)
data_scaledMMf=pd.DataFrame(data_MMf)
data_scaledMMf.columns=col_namesf
data_scaledMMf.head()


# ##### Normalization

# - Se usa también un escalamiento de los datos con la normalización, restando la media y dividiendo en su desviación estándar. Este tipo de escalamiento no se realiza en la aproximación de One Hot Encoding, pues la media de una variable dummy es la proporción de 1, no su media en si.

# In[488]:


scalerN=StandardScaler()
data_Nf = scalerN.fit_transform(data_freq)
data_scaledNf=pd.DataFrame(data_Nf)
data_scaledNf.columns=col_namesf
data_scaledNf.head()


# ### PCA

# - Se propone la utilización de componentes principales con el ánimo de capturar la mayor cantidad de variación de múltiples variables en un sólo factor y reducir así la dimensionalidad. El problema de esto, es la interpretación de los PCA, pues ya no será tan simple cómo con las variables normalmente. Para realizar PCA se usan los valores escalados para que las diferentes medidas de unidades de las variables no afecte las combinaciones lienales de los factores.

# In[489]:


from sklearn.decomposition import PCA


# - Se obtiene PCA para los datos escalados por MinMAX para el dataframe que se le aplicó OneHot Encoding.
# - Se gráfica la cantidad de varianza explicada por la cantida de factores, la idea es obtener un pareto, tratar de obtener el 80% de la variación explicada por los componentes. Esto ocurre entre 10 y 20 componenetes.

# In[490]:


pcaMMd = PCA().fit(data_scaledMMd)
plt.plot(np.cumsum(pcaMMd.explained_variance_ratio_))
plt.xlabel('número de componentes')
plt.ylabel('% varianza explicada acumulada');


# - Se apoya la elección del número de componentes con el 'gráfico de codo', donde haya un cambio de pendiente en esta curva se elige este número de componentes. Note que los primeros componentes capturan el grueso de la variabilidad, a más componentes menos variabilidad capturan en términos marginales.

# In[491]:


features = range(pcaMMd.n_components_)
plt.bar(features, pcaMMd.explained_variance_ratio_, color='black')
plt.xlabel('Componentes PCA')
plt.ylabel('varianza %')


# - Dados los gráficos expuesto antes se elegirán 10 componentes.

# In[492]:


pcaMMd = PCA(n_components=10) 
principalcomponentsMMd = pcaMMd.fit_transform(data_scaledMMd)
data_pcaMMd = pd.DataFrame(principalcomponentsMMd)
data_pcaMMd=data_pcaMMd.add_prefix('pca_')
data_pcaMMd.head()


# - Se realiza también para el data frame escalado con Frequency encoding.
# - Los criterios de elección de lso componnetes siguen la misma lógica que lso descritos antes, usando las gráficas, por tal motivo no se explicará una y otra vez para los diferentes dataframes, con el ánimo de no ser redundantes.

# In[493]:


pcaMMf = PCA().fit(data_scaledMMf)
plt.plot(np.cumsum(pcaMMf.explained_variance_ratio_))
plt.xlabel('número de componentes')
plt.ylabel('% varianza explicada acumulada');


# In[494]:


features = range(pcaMMf.n_components_)
plt.bar(features, pcaMMf.explained_variance_ratio_, color='black')
plt.xlabel('Componentes PCA')
plt.ylabel('varianza %')


# In[495]:


pcaMMf = PCA(n_components=6) 
principalcomponentsMMf = pcaMMf.fit_transform(data_scaledMMf)
data_pcaMMf = pd.DataFrame(principalcomponentsMMf)
data_pcaMMf=data_pcaMMf.add_prefix('pca_')
data_pcaMMf.head()


# - Se realiza PA para datos normalizados bajo frequency encoding.

# In[496]:


pcaNf = PCA().fit(data_scaledNf)
plt.plot(np.cumsum(pcaNf.explained_variance_ratio_))
plt.xlabel('número de componentes')
plt.ylabel('% varianza explicada acumulada');


# In[497]:


features = range(pcaNf.n_components_)
plt.bar(features, pcaNf.explained_variance_ratio_, color='black')
plt.xlabel('Componentes PCA')
plt.ylabel('varianza %')


# In[498]:


pcaNf = PCA(n_components=4) 
principalcomponentsNf = pcaNf.fit_transform(data_scaledNf)
data_pcaNf = pd.DataFrame(principalcomponentsNf)
data_pcaNf=data_pcaNf.add_prefix('pca_')
data_pcaNf.head()


# # MODELADO-Clasificación

# - Cómo se cuenta con diferentes datasets dependiendo del 'Encoding', del escalamiento y de si se les aplicó PCA. Se realizarán varias particiones de los datos para ser pasadas a los diferentes algoritmos de clasificación elegidos.

# - Variables independientes de la aproximación OneHot Encoding.

# In[499]:


xd=data_dummies.values


# - Variables independientes de la aproximación frequency Encoding.

# In[500]:


xf=data_freq.values


# - Variables independientes de la aproximación escalación MinMax OneHot Encoding.

# In[501]:


xscMd=data_scaledMMd.values


# - Variables independientes de la aproximación escalación MinMax frequency Encoding.

# In[502]:


xscMf=data_scaledMMf.values


# - Variables independientes de la aproximación escalación normalización frequency Encoding.

# In[503]:


xscNf=data_scaledNf.values


# - Variables independientes de la aproximación PCA escalación MinMax OneHot Encoding.

# In[504]:


xpcaMd=data_pcaMMd.values


# - Variables independientes de la aproximación PCA escalación MinMax frequency Encoding.

# In[505]:


xpcaMf=data_pcaMMf.values


# - Variables independientes de la aproximación PCA escalación normalización frequency Encoding.

# In[506]:


xpcaNf=data_pcaNf.values


# - Distribución de la variable dependiente, esta será la misma para todas las aproximaciones propuestas arriba

# In[507]:


print(data['readmitted'].values)


# In[508]:


y=data.readmitted.values


# - Se codifica las categorías de la variable dependiente con enteros.

# In[509]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y


# In[510]:


list(labelencoder.classes_)


# In[ ]:





# In[511]:


x=data.drop(columns=['readmitted'])


# - Partición sets de entrenamiento y prueba de acuerdo a las aproximaciones propuestas

# In[512]:


from sklearn import model_selection


# -  Aproximación OneHot Encoding.

# In[513]:


x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(xd, y, test_size=0.2,stratify=y)


# - Aproximación frequency Encoding.

# In[514]:


x_train2, x_test2, y_train2, y_test2 = model_selection.train_test_split(xf, y, test_size=0.2,stratify=y)


# - Aproximación escalación MinMax OneHot Encoding.

# In[515]:


x_train3, x_test3, y_train3, y_test3 = model_selection.train_test_split(xscMd, y, test_size=0.2,stratify=y)


# - Aproximación escalación MinMax Frequency Encoding.

# In[516]:


x_train4, x_test4, y_train4, y_test4 = model_selection.train_test_split(xscMf, y, test_size=0.2,stratify=y)


# - Aproximación escalación Normalización Frequency Encoding.

# In[517]:


x_train5, x_test5, y_train5, y_test5 = model_selection.train_test_split(xscNf, y, test_size=0.2,stratify=y)


# - Aproximación PCA escalación MinMax OneHot Encoding.

# In[518]:


x_train6, x_test6, y_train6, y_test6 = model_selection.train_test_split(xpcaMd, y, test_size=0.2,stratify=y)


# - Aproximación PCA escalación MinMax Frequency Encoding.

# In[519]:


x_train7, x_test7, y_train7, y_test7 = model_selection.train_test_split(xpcaMf, y, test_size=0.2,stratify=y)


# - Aproximación PCA escalación normalización Frequency Encoding.

# In[520]:


x_train8, x_test8, y_train8, y_test8 = model_selection.train_test_split(xpcaNf, y, test_size=0.2,stratify=y)


# ## Modelado

# - Cada una de las corridas de los algoritmos empleando las diferentes particiones propuestas se divide por números del 1 al 8, y se encuentran en el mismo orden al presentado arriba.

# In[521]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


# - Árbol de clasificación

# - 1

# In[522]:


ct1 = DecisionTreeClassifier(criterion='gini')
ct1.fit(x_train1,y_train1)
y_predct1=ct1.predict(x_test1)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test1, y_predct1)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test1,y_predct1)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test1, y_predct1))


# - 2

# In[523]:


ct2 = DecisionTreeClassifier(criterion='gini')
ct2.fit(x_train2,y_train2)
y_predct2=ct2.predict(x_test2)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test2, y_predct2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_predct2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_predct2))


# In[577]:


feature_importancetrees=pd.Series(ct2.feature_importances_,index=x.columns)
feature_importancetrees.nlargest(30).plot(kind='barh', figsize=(10,8),grid=False)
plt.title('Importancia de las variables- Decission Trees Classifier')
plt.show()


# - Se gráfica la importancia de las variables en este ajuste basándose en el criterio que es el modelo cuyo Accuracy es mayor (existe empate con otros modelos, pero la interpretación es más fácil, ya que no se trata de variables binarias o de PCA, son las mismas variables sólo que codificadas bajo en enfoque de Frequency encoding). 

# - Observe que las 5 variables más relevantes fueron: glipzide_metformin, metformin-rosiglitazone, glyburide-metformin, age, glimepiride-pioglitazone, cuatro de ellas son medicamentos para tartaar la diabetes.

# - 3

# In[524]:


ct3 = DecisionTreeClassifier(criterion='gini')
ct3.fit(x_train3,y_train3)
y_predct3=ct3.predict(x_test3)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test3, y_predct3)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test3,y_predct3)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test3, y_predct3))


# - 4

# In[525]:


ct4 = DecisionTreeClassifier(criterion='gini')
ct4.fit(x_train4,y_train4)
y_predct4=ct4.predict(x_test4)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test4, y_predct4)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test4,y_predct4)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test4, y_predct4))


# - 5

# In[526]:


ct5 = DecisionTreeClassifier(criterion='gini')
ct5.fit(x_train5,y_train5)
y_predct5=ct5.predict(x_test5)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test5, y_predct5)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test5,y_predct5)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test5, y_predct5))


# - 6

# In[527]:


ct6 = DecisionTreeClassifier(criterion='gini')
ct6.fit(x_train6,y_train6)
y_predct6=ct6.predict(x_test6)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test6, y_predct6)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test6,y_predct6)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test6, y_predct6))


# - 7

# In[528]:


ct7 = DecisionTreeClassifier(criterion='gini')
ct7.fit(x_train7,y_train7)
y_predct7=ct7.predict(x_test7)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test7, y_predct7)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test7,y_predct7)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test7, y_predct7))


# - 8

# In[529]:


ct8 = DecisionTreeClassifier(criterion='gini')
ct8.fit(x_train8,y_train8)
y_predct8=ct8.predict(x_test8)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test8, y_predct8)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test8,y_predct8)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test8, y_predct8))


# ##### Conclusiones trees

# - Para este caso, los áboles no presentaron un buen desempeño. Fue muy bajo, a pesar que los árboles normalmente reciben variables categóricas, para el .fit() de Skelarn se debe de pasar variables numéricas. Quiza las codificaciones de las variables, como el OneHotEncoding generaron muchas variables y no permitieron una buena clasificación. No obstante se pensó que con la aproximación de Frequency encoding, que no genera tantas variables como OneHot, tendría un mejor desempeño. Las 5 variables más relevantes fueron: glipzide_metformin, metformin-rosiglitazone, glyburide-metformin, age, glimepiride-pioglitazone, todas menos age, fueron medicinas génericas para tratar la diabetes.
# - Los árboles tienen el problema de tener alta varianza, se puede corregir con Random forest.

# ### Regresión logística

# In[530]:


from sklearn.linear_model import LogisticRegression


# - Se usa regularización combinada L1 y L2 elastic net para controlar por el gran número de variables.

# - 1

# In[531]:


log1=LogisticRegression(random_state=123, penalty="elasticnet", solver='saga',l1_ratio=0.5)
log1.fit(x_train1, y_train1)
y_predlog1=log1.predict(x_test1)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test1, y_predlog1)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test1,y_predlog1)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test1, y_predlog1))


# - 2

# In[532]:


log2=LogisticRegression(random_state=123, penalty="elasticnet", solver='saga',l1_ratio=0.5)
log2.fit(x_train2, y_train2)
y_predlog2=log2.predict(x_test2)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test2, y_predlog2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_predlog2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_predlog2))


# - 3

# In[533]:


log3=LogisticRegression(random_state=123, penalty="elasticnet", solver='saga',l1_ratio=0.5)
log3.fit(x_train3, y_train3)
y_predlog3=log3.predict(x_test3)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test3, y_predlog3)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test3,y_predlog3)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test3, y_predlog3))


# - 4

# In[534]:


log4=LogisticRegression(random_state=123, penalty="elasticnet", solver='saga',l1_ratio=0.5)
log4.fit(x_train4, y_train4)
y_predlog4=log4.predict(x_test4)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test4, y_predlog4)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test4,y_predlog4)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test4, y_predlog4))


# - 5

# In[535]:


log5=LogisticRegression(random_state=123, penalty="elasticnet", solver='saga',l1_ratio=0.5)
log5.fit(x_train5, y_train5)
y_predlog5=log5.predict(x_test5)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test5, y_predlog5)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test5,y_predlog5)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test5, y_predlog5))


# - 6

# In[536]:


log6=LogisticRegression(random_state=123, solver='saga')
log6.fit(x_train6, y_train6)
y_predlog6=log6.predict(x_test6)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test6, y_predlog6)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test6,y_predlog6)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test6, y_predlog6))


# - 7

# In[537]:


log7=LogisticRegression(random_state=123, solver='saga')
log7.fit(x_train7, y_train7)
y_predlog7=log7.predict(x_test7)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test7, y_predlog7)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test7,y_predlog7)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test7, y_predlog7))


# - 8

# In[538]:


log8=LogisticRegression(random_state=123, solver='saga')
log8.fit(x_train8, y_train8)
y_predlog8=log8.predict(x_test8)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test8, y_predlog8)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test8,y_predlog8)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test8, y_predlog8))


# ##### Conclusión logit

# - Se obtuvieron mejores desempñeos que con árboles, esta fue la aproximación del paper, salvo que en este se trato de un problema de clasificación binario, en eset caso sería logit multiclase.

# ## Random Forest

# In[539]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


# - 1

# In[540]:


rf1 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf1.fit(x_train1,y_train1)
y_predrf1=rf1.predict(x_test1)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test1, y_predrf1)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test1,y_predrf1)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test1, y_predrf1))


# - 2

# In[541]:


rf2 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf2.fit(x_train2,y_train2)
y_predrf2=rf2.predict(x_test2)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test2, y_predrf2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_predrf2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_predrf2))


# - Se gráfica la importancia de las variables en este ajuste basándose en el criterio que es el modelo cuyo Accuracy es mayor (existe empate con otros modelos, pero la interpretación es más fácil, ya que no se trata de variables binarias o de PCA, son las mismas variables sólo que codificadas bajo en enfoque de Frequency encoding). 

# In[580]:


feature_importancerf=pd.Series(rf2.feature_importances_,index=x.columns)
feature_importancerf.nlargest(30).plot(kind='barh', figsize=(10,8),grid=False)
plt.title('Importancia de las variables- Random Forest')
plt.show()


# - Note que al igual que en el algortimo de árboles de clasificación, las variables más relevantes fueron medicinas: glipzide-metformin, metformin-rosiglitazone, glyburide-metformin, age, diagnose_1, salvo que las dos últimas está age y el diagnóstico.

# - 3

# In[542]:


rf3 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf3.fit(x_train3,y_train3)
y_predrf3=rf3.predict(x_test3)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test3, y_predrf3)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test3,y_predrf3)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test3, y_predrf3))


# - 4

# In[543]:


rf4 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf4.fit(x_train4,y_train4)
y_predrf4=rf4.predict(x_test4)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test4, y_predrf4)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test4,y_predrf4)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test4, y_predrf4))


# - 5

# In[544]:


rf5 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf5.fit(x_train5,y_train5)
y_predrf5=rf5.predict(x_test5)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test5, y_predrf5)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test5,y_predrf5)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test5, y_predrf5))


# - 6

# In[545]:


rf6 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf6.fit(x_train6,y_train6)
y_predrf6=rf6.predict(x_test6)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test6, y_predrf6)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test6,y_predrf6)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test6, y_predrf6))


# - 7

# In[546]:


rf7 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf7.fit(x_train7,y_train7)
y_predrf7=rf7.predict(x_test7)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test7, y_predrf7)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test7,y_predrf7)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test7, y_predrf7))


# - 8

# In[547]:


rf8 = RandomForestClassifier(n_estimators=200,verbose=1,min_samples_leaf=5)
rf8.fit(x_train8,y_train8)
y_predrf8=rf8.predict(x_test8)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test8, y_predrf8)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test8,y_predrf8)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test8, y_predrf8))


# ##### Conclusiones Random Forest

# - Se obtuvo desempñeo igual que logit y mejor que árboles. Se espera que el uso de estos métodos de remuestreo de árboles mejore la varianza y la estimación. Las variables más relevantes coinciden con las de árboles, salvo que se introduce el diagnóstico.

# ### Boosting Adaboost

# - Se usa Meta algoritmo Boosting con Adaboost

# - 1

# In[548]:


boosting1 = AdaBoostClassifier(n_estimators=200)
boosting1.fit(x_train1,y_train1)
y_predboosting1 = boosting1.predict(x_test1)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test1, y_predboosting1)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test1,y_predboosting1)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test1, y_predboosting1))


# - 2

# In[549]:


boosting2 = AdaBoostClassifier(n_estimators=200)
boosting2.fit(x_train2,y_train2)
y_predboosting2 = boosting2.predict(x_test2)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test2, y_predboosting2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_predboosting2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_predboosting2))


# In[ ]:





# In[581]:


feature_importanceboost=pd.Series(boosting2.feature_importances_,index=x.columns)
feature_importanceboost.nlargest(30).plot(kind='barh', figsize=(10,8),grid=False)
plt.title('Importancia de las variables- Adaboost')
plt.show()


# - Las variables más relevantes fueron: glipzide-metformin, metformin-rosiglitazone, metformin-pioglitazone, age, diabtesMed, las medicinas para el tratamiento de la enfermedad, la edad y la Dummy relacionada con si tiene prescripción de medicina para diabéticos.

# - 3

# In[550]:


boosting3 = AdaBoostClassifier(n_estimators=200)
boosting3.fit(x_train3,y_train3)
y_predboosting3 = boosting3.predict(x_test3)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test3, y_predboosting2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_predboosting2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_predboosting2))


# - 4

# In[551]:


boosting4 = AdaBoostClassifier(n_estimators=200)
boosting4.fit(x_train4,y_train4)
y_predboosting4 = boosting4.predict(x_test4)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test4, y_predboosting4)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test4,y_predboosting4)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test4, y_predboosting4))


# - 5

# In[552]:


boosting5 = AdaBoostClassifier(n_estimators=200)
boosting5.fit(x_train5,y_train5)
y_predboosting5 = boosting5.predict(x_test5)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test5, y_predboosting5)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test5,y_predboosting5)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test5, y_predboosting5))


# - 6

# In[553]:


boosting6 = AdaBoostClassifier(n_estimators=200)
boosting6.fit(x_train6,y_train6)
y_predboosting6 = boosting6.predict(x_test6)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test6, y_predboosting6)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test6,y_predboosting6)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test6, y_predboosting6))


# - 7

# In[554]:


boosting7 = AdaBoostClassifier(n_estimators=200)
boosting7.fit(x_train7,y_train7)
y_predboosting7 = boosting7.predict(x_test7)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test7, y_predboosting7)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test7,y_predboosting7)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test7, y_predboosting7))


# - 8

# In[555]:


boosting8 = AdaBoostClassifier(n_estimators=200)
boosting8.fit(x_train8,y_train8)
y_predboosting8 = boosting8.predict(x_test8)
#Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test8, y_predboosting8)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test8,y_predboosting8)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test8, y_predboosting8))


# ##### Conclusiones Boosting

# - Se obtienen un desempñeo igual al random Forest, las variables más relevantes vuelven a ser los medicamnetos para el tratamiento de esta enfermedad, la edad y la dummy de si se medica para diabetes, la cual se correlaciona con el hecho que tome cierta medicina genérica para diabetes.

# ## Naive Bayes

# - Este clasificador puede tener problemas pues asume independencia de los atributos, y posiblemente no exista dicha independencia.

# In[557]:


from sklearn.naive_bayes import GaussianNB
nb1 = GaussianNB()
nb1.fit(x_train1, y_train1)
y_prednb1 = nb1.predict(x_test1)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test1, y_prednb1)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test1,y_prednb1)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test1, y_prednb1))


# - Es pésimo su desempeño.
# - Se prueba con Frquency encoding.

# In[558]:


nb2 = GaussianNB()
nb2.fit(x_train2, y_train2)
y_prednb2 = nb2.predict(x_test2)
# Accuracy y matriz de confusión
accuracy = metrics.accuracy_score(y_test2, y_prednb2)
print("Accuracy: {:.2f}".format(accuracy))
cm=confusion_matrix(y_test2,y_prednb2)
print('Confusion Matrix: \n', cm)
print(classification_report(y_test2, y_prednb2))


# ##### Conclusiones Naive Bayes

# - Se obtiene el peor de lso resultados, posiblemente el supuesto de independencia en los atributos sea muy fuerte en este caso, donde los atributos están muy correlacionados.

# In[560]:


import gc
gc.collect()


# ## Light GBM

# - Permite pasar las variables categóricas sin antes voilverlas numéricas

# In[561]:


import lightgbm as lgb


# In[562]:


data.info()


# In[563]:


x = data.drop(columns = ['readmitted'])


# In[564]:


for c in x.columns:
    col_type = x[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        x[c] = x[c].astype('category')


# In[565]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)


# In[566]:


cflgb = lgb.LGBMClassifier(loss_function= 'Logloss', custom_metric=['Accuracy'],eval_metric='F1')
cflgb.fit(x_train, y_train, eval_set=(x_test, y_test), feature_name='auto', categorical_feature = 'auto', verbose=50)
print(); print(cflgb)
y_predlgb = cflgb.predict(x_test)
print(); print(metrics.classification_report(y_test, y_predlgb))
print(); print(metrics.confusion_matrix(y_test, y_predlgb))


# In[567]:


feature_importance=pd.Series(cflgb.feature_importances_,index=x.columns)
feature_importance.nlargest(30).plot(kind='barh', figsize=(10,8),grid=False)
plt.title('Importancia de las variables- LightGBM')
plt.show()


# - Las variables más relevantes fueron numéricas a digferencia de los otros algortimos, indicando que el número de diagnósticos, procedimientos, pruebas, edad y el tiempo en el hospital explican la clasificación de la variable dependiente.

# ##### Conclusiones Light GBM

# - Este algoritmo es interesante pues permite introducir las variables categóricas sin realizarse un preprocesamiento previo que pueda llegar a distorsionar la relación de estas variable so a generar problemas muy dispersos por la alta dimensionalidad. Dejando de lado esto, indica que las variables más relevantes son númericas que pueden ser proxies de que tan malo es el estado del paciente, pues mayor tiempo de estancia en el hospital, más estudios y pruebas y una edad avanzada pueden hacer que la persona vuelva al hospital.

# ### CatBoost

# In[568]:


from catboost import CatBoostClassifier


# In[569]:


data_cat1=data_cat.drop(['readmitted'],axis=1)
data1=data.drop(['readmitted'],axis=1)


# In[570]:


categorical_names=data_cat1.columns.tolist()
print(categorical_names)


# In[571]:


categoricals = [data1.columns.get_loc(i) for i in categorical_names]
print(categoricals)


# In[572]:


from sklearn.multiclass import OneVsRestClassifier


# In[573]:


ovr = OneVsRestClassifier(estimator=CatBoostClassifier(iterations=10,random_state=123, cat_features=categoricals, loss_function= 'Logloss', custom_metric=['Accuracy'], eval_metric='F1'))
ovr.fit(x_train,y_train)
y_predovrcatboost=ovr.predict(x_test)
print(); print(ovr)
print(); print(metrics.classification_report(y_test, y_predovrcatboost))
print(); print(metrics.confusion_matrix(y_test, y_predovrcatboost))


# ##### Conclusiones Catboost

# - El desempñeo de este modelo es igual al de Random Forest, AdaBoost, entre otros. El boost sobre categorias sólo está pensado para una clasificación de dos clases, por tal motivo debió emplearse una clasificación one vs de rest, dond eprimero se predecia la categoria 1 contra las otra sdos agrupadas como el resto, luego se hace esto mismo para la categoría dos y finalmente para la categoria tres.

# # INSIGHTS Y CONCLUSIONES

# - Se emplearon varios clasificadores y el desempñeo de clasificación multiclase no es tan bueno, es un poco por encima del chance.
# - Se contaba con un problema donde había muchas variables categóricas que necesitaban ser tratadas y preprocesadas. Por tal motivo, se recurrió a los Encodings (OneHOt, Frequency), a la reducción de dimensionalidad mediante PCA. Estos tratamientos pueden generar problemas en las estimaciones al llevar a problemas de alta dimensionalidad e insertar mucha complejidad al modelo.
# - Se encuentra que en casi todos los algoritmos, las variables relavantes son similares, haciendo alución a medicamentos genéricos para tratar la diabetes, también resalta la edad. Al parecer la edad (ser más viejo o no) afecta si reingreso o no, lo cual tiene sentido, al suponer que una edad avanzada significan más quebrantos de salud. Asimismo, otra variable importante fue el diagnóstico, lo cual también se relaciona con el hecho que diagnósticos más graves posiblemente influirán en si reingreso o no. Con la implementanción del light GBM se pudo pasar los datos categóricos como tal sin necesidad de codificarlos y se obtuvo que las varaibles relevantes eran numéricas relacionadas con el tiempo de estancia en el hospital, la cantidad de procedimientos y pruebas, y la edad. Todas estan pueden asociarse como proxy de un mal estado o de quebrantos de salud, que llevan a que se le realicen más estudios y pruebas al paciente y que su condición se agrave lo cual posiblemnete incida en que vuelva o no a ser admitido en un futuro.
# - Se puede realizar mayor trabajo con el feature engineering, seleccionando otras variables o generando interacciones entre variables que resultaron relevantes, paar esto se necesita un backgroung médico que permita crear las interacciones pertinentes entre esta svariables.
# - Puede asimismo pensarse en la posibilidad de realziar una estimación one VS de rest para volver este problema de clasificación binario y probarlo sobre los algoritmos. Esto sólo se hizo sobre el CatBoost debido a que sólo podía hacerse sobre escenarios de clasificación de dos clases.
# - Puede pensarse en que hacer con los outliers de algunas variables, si la imputación de ciertos valores haría que perdiera variabilidad en dicho atributo (dado que casi toda la variable estaba representada por un valor y los únicos qu ele daban variabilidad a esta seran los outliers) o no y su impacto en la estimación.
# - Podría pensarse que hay un problema de desbalance de clases en las categorías, como pudo observarse, la categoría <30 días para readmisión tiene muy pocas observaciones, podría tratarse a futuro, con sobremuestreo que nivele las clases y junto la implementación de meta algoritmos como Boosting que permite ir sopesando por pesos las clasificaciones de manera secuencial.
# - Podría realizarse un hyper parameter tunning. El cual podria brindar la elección de parámetrso óptimos qu epermitan un mejor desempñeo. Esta aproximación se desistió porque se contaba con muchas aproximaciones de diferentes encodings además de modelos.
