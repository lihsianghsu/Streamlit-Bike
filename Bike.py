import streamlit as st
import geopandas as gpd
import seaborn as sns
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px
from streamlit_option_menu import option_menu
#from numerize.numerize import numerize
import folium
from streamlit_folium import folium_static
#from streamlit_folium import st_folium
import joblib
from joblib import load
#import pickle5 as pickle
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import plost
from sklearn.linear_model import LinearRegression, RidgeCV
#import pandas_profiling
#from query import *
from time import time

st.set_page_config(page_title="Comptage de vélo à Paris",page_icon="🌍",layout="wide")
st.image('graph_00.png')
st.write("<h4 style=text-align:center;>Analyse des données de trafic des 🚴 à Paris</h4>", unsafe_allow_html=True)

#st.markdown("##")

theme_plotly = None # None or streamlit

#Créer la page Web pour comptage vélo
st.sidebar.header("Paris, ville 100 % cyclable 🚲")
st.sidebar.image("compteurs_de_vélo_01.jpg",caption="Developed and Maintaned by: Mohamed Arrar, Wilfried Dartois, Li-Hsiang Hsu")



pages=["À propos", 'Projet', 'Jeux de données', 'DataViz', 'Modélisation', 'Entraînez votre modèle : À vous de jouer !', 'Interprétabilité', 'Bilan et perspective']
with st.sidebar:
    selected = option_menu("Menu principal", pages, icons=['house'], menu_icon="cast", default_index=1)
    

st.sidebar.header("🚴‍♀️Faire du vélo à Paris🚴‍♀️")


if selected == 'À propos' : 
    st.write("")
    st.write("<h1 style=text-align:center;>Projet</h1>", unsafe_allow_html=True)
    st.write("<h1 style=text-align:center;>Comptage de vélo à Paris</h1>", unsafe_allow_html=True)
    st.write("<h2 style=text-align:center;>Formation Data Analyst de DataScientest.com</h2>", unsafe_allow_html=True)
    st.write("<h3 style=text-align:center;>du janvier au septembre 2023</h3>", unsafe_allow_html=True)
    st.write("<h1 style=text-align:center;>Auteurs:</h1>", unsafe_allow_html=True)
    st.write("<h2 style=text-align:center;>Mohamed Arrar</h2>", unsafe_allow_html=True)
    st.write("<h2 style=text-align:center;>Wilfried Dartois</h2>", unsafe_allow_html=True)
    st.write("<h2 style=text-align:center;>Li-Hsiang Hsu</h2>", unsafe_allow_html=True)

    #st.write("<h4 style=text-align:center;>Analyse des données de trafic 🚴 à Paris</h4>", unsafe_allow_html=True)

elif selected == 'Projet':
    st.write("")
    #st.image('compteurs_de_vélo.jpg')
    # Plan de Paris
    #m = folium.Map(location = [48.856578, 2.351828], zoom_start = 12)
    #folium.Marker(location=[48.856578, 2.351828]).add_to(m)
    #plan_paris = st_folium(m, width=725)
    #Description du projet
    st.write("<h2 style=text-align:center;>Projet</h2>", unsafe_allow_html=True)
    st.write("Il s'agit d'un rapport d'exploration, de visualisation de données et de pré-traitement pour un projet de la ville de Paris qui vise à utiliser les données provenant de compteurs de vélo permanents pour identifier les zones de forte circulation pendant des périodes spécifiques nécessitant différents besoins en termes d'infrastructures, de rénovation ou de sécurité. ")
    st.write("L'objectif est de comprendre l'évolution du trafic cycliste en fonction des secteurs de la ville et des périodes clés, ainsi que les facteurs qui influencent le choix entre le vélo, la voiture et les transports en commun.")
    st.write("")
    st.write("")
    if st.checkbox("Afficher les bibliothèques utilisées") :
        st.write("Bibliothèques utilisées")
        st.image('biblio.png')
elif selected == "Jeux de données":
    st.write("")
    st.write("<h2 style=text-align:center;>Jeux de données et pré-traitement</h2>", unsafe_allow_html=True)
    
     
    st.write("")
    #st.write("<h3 style=text-align:center;>Pré-traitement</h3>", unsafe_allow_html=True)
    
    st.write("")
    st.image('Pre_processing.jpg')
    st.write("<h5 style=text-align:center;>Les étapes de pré-traitement</h5>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    if st.checkbox("Afficher les étapes détailléees:") :
        st.write("")
        st.write("➢	Ajout/Modification de variables au jeu de données principal")
        st.write("• Périodicité : Month, Weekday, Weeknum, Year, Day…")
        st.write("•	Evènement dans la ville (Grève, Manifestation gilets jaunes, pénurie d’essence)")
        st.write("•	Données géographique : Arrondissement, Longitude, Latitude")
        st.write("•	Informations météorologique : Opinion_weather")
        st.write("➢	Réorganisation des colonnes et suppression des colonnes manquantes")
        st.write("➢	Filtre des données sur une année complète")
        st.write("➢	Traitement des données")
        st.write("•	Valeurs manquante : Les valeurs manquantes du fichier n’ont aucun impact sur notre travail car elles se trouvent dans des colonnes que nous n’utilisons pas.")
        st.write("•	Outliers : Nous avions constaté grâce à une analyse sur les outliers une erreur qui en approfondissement sont dues à une panne du compteur localisé au « 39 quai François Mauriac » sur le mois d’août uniquement. Nous avons donc fait le choix d’extraire de notre Dataframe les données cités ci-dessus.")
        st.write("")
        st.image('outliers.png')
        st.write("")
        st.write("")
    st.write("<h2 style=text-align:center;>Explorez nos jeux de données en profondeur</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")    
    choix_df = ['Rien, merci !',
                'Sources de données', 
                'Jeu de données pour DataViz', 
                'Jeux de données pour modélisation']
    disply_df = st.radio('Que souhaitez-vous explorer ?', choix_df)
    

    if disply_df == 'Non, merci!':
        st.write("")
        
    if disply_df == 'Sources de données':
        st.write("Vous avez choisi : Sources de données")
        #st.image('Viz_00.png')
        st.write("<h5 style=text-align:center;>Sources de données :</h5>", unsafe_allow_html=True)

        link1 = "https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name"
        text1 = "Comptage vélo: Données compteurs de la Ville de Paris"
        link2 = "https://www.historique-meteo.net/"
        text2 = "Données météo : Historique Météo"
        st.write(f'<h5 style="text-align:center;"><a href="{link1}" target="_blank">{text1}</a></h5>', unsafe_allow_html=True)
        st.write(f'<h5 style="text-align:center;"><a href="{link2}" target="_blank">{text2}</a></h5>', unsafe_allow_html=True)
        st.write("")
    
        st.write("<h3 style=;>➢	Données de comptage de vélo :</h3>", unsafe_allow_html=True)
        st.write("Les données exploitées sont en open data et donc libre accès pour une utilisation appropriée. Le poids de notre jeu de données est de 1,15 Go, comprenant 16 colonnes et 724 467 lignes.")
        st.write("Les variables les plus pertinentes pour nos objectifs incluent le nom du compteur, la date et l'heure de comptage en tant que variable temporelle, la date d'installation du site de comptage pour expliquer la différence de la variable de comptage sur une période longue, les coordonnées géographiques pour la localisation précise sur une cartographie et la réalisation de statistiques de moyenne par arrondissement, et le mois et l'année du comptage pour le regroupement des valeurs.")
        st.write("La variable cible de notre projet est le comptage horaire.")
        st.write("Afin d’appréhender les enjeux et objectifs de notre projet nous avons intégré un ensemble de jeux de données qui sont les suivants : ")
        st.write("<h3 style=;>➢	Météo de Paris sur l’année 2022 :</h3>", unsafe_allow_html=True)
        st.write("L’objectif de ce fichier était de nous permettre d’obtenir la météo sur la ville pour chaque jour de l’année 2022. Cela va nous permettre d’identifier son impact sur notre jeux de données de base.")
        st.write("<h3 style=;>➢	Arrondissement :</h3>", unsafe_allow_html=True)
        st.write("Le fichier contenant les arrondissement nous a permis de situer sur une carte chaque compteur et de les regrouper par arrondissement ce qui permet de faire une analyse en fonction de l’utilisation des vélos pour chaque zone de la ville de Paris")
        st.write("<h3 style=;>➢	Événement en Ile De France : :</h3>", unsafe_allow_html=True)
        st.write("Ce fichier regroupe toutes les grèves de transport de la RATP, Manifestations sociales et pénurie d’essence de l’année 2022. Nous pourrons donc voir comme pour le fichier de la météo l’impact que chaque évènement aura eu sur l’utilisation d’un vélo comme moyen de transport.")
        
    if disply_df == 'Jeu de données pour DataViz':
        st.write("Vous avez choisi : Jeu de données pour DataViz")
        st.write("<h3 style=;>Jeu de données pour DataViz :</h3>", unsafe_allow_html=True)
        df_viz = pd.read_csv('df_viz.csv')
        st.dataframe(df_viz.head(20), hide_index=True)
        
        if st.checkbox("Afficher le résumé descriptif") :
            st.write("")
            st.write("<h3 style=;>Résumé descriptif :</h3>", unsafe_allow_html=True)
            st.write("")
            st.dataframe(df_viz.describe())
            st.write("")
    #st.write("<h3 style=;>Data Profiling :</h3>", unsafe_allow_html=True)
    #from streamlit_pandas_profiling import st_profile_report
    #import ydata_profiling
    #from ydata_profiling import ProfileReport

    #pr = df_viz.profile_report()
    #st_profile_report(pr)
    #pr = ProfileReport(df_viz, title="Pandas Profiling Report")
    #pr
        if st.checkbox("Afficher les Informations des colonnes") :
            st.write("")
            st.write("<h3 style=;>Informations des colonnes :</h3>", unsafe_allow_html=True)
            st.write("")
            import io
            # Capture the output of info()
            info_buffer = io.StringIO()
            df_viz.info(buf=info_buffer)
            info_text_viz = info_buffer.getvalue()
            # Display the output using st.text()
            st.text(info_text_viz)
            st.write("")
            
    if disply_df == 'Jeux de données pour modélisation':
        st.write("")
        st.write("Vous avez choisi : Jeux de données pour modélisation")
        st.write("<h3 style=;>Jeux de données pour modélisation: </h3>", unsafe_allow_html=True)
        #X_train = pd.read_csv('X_train.csv')
        X_train = pd.read_csv('X_train.csv')
        y_train = pd.read_csv('y_train.csv')
        X_test = pd.read_csv('X_test.csv')
        st.write("Concernant l’étape de modélisation nous faisons le parti pris de supprimer certaines données et d’en conserver qu’un nombre défini afin d’avoir une pertinence et un résultat plus affiné.")
        st.write("Etape de préparation des données à la modélisation:")
        st.write("➢ Sélection des colonnes à conserver pour la modélisation")
        st.write("➢ Répartition des données en Test et Train")
        st.write("➢ Standardisation des données")
        st.write("")
        st.write("")
        st.write("Concernant ce choix nous décidons de conserver les données suivantes : ")
        st.dataframe(X_train.head(), hide_index=True)
        st.write("")
    
        if st.checkbox("Afficher le résumé descriptif des données pour modélisation (entraînement)") :
            st.write("")
            st.write("<h3 style=;>Résumé descriptif (entraînement):</h3>", unsafe_allow_html=True)
            st.write("")
            st.dataframe(X_train.describe())
            st.write("")

        if st.checkbox("Afficher les Informations des colonnes (entraînement)") :
            st.write("")
            st.write("<h3 style=;>Informations des colonnes (entraînement):</h3>", unsafe_allow_html=True)
            st.write("")
            import io
            # Capture the output of info()
            info_buffer = io.StringIO()
            X_train.info(buf=info_buffer)
            info_text_train = info_buffer.getvalue()
            # Display the output using st.text()
            st.text(info_text_train)
            st.write("")    

        if st.checkbox("Afficher le résumé descriptif des données pour modélisation (test)") :
            st.write("")
            st.write("<h3 style=;>Résumé descriptif (test) :</h3>", unsafe_allow_html=True)
            st.write("")
            st.dataframe(X_test.describe())
            st.write("")
        
        if st.checkbox("Afficher les Informations des colonnes (test)") :
            st.write("")
            st.write("<h3 style=;>Informations des colonnes (test):</h3>", unsafe_allow_html=True)
            st.write("")
            import io
            # Capture the output of info()
            info_buffer = io.StringIO()
            X_test.info(buf=info_buffer)
            info_text_test = info_buffer.getvalue()
            # Display the output using st.text()
            st.text(info_text_test)
            st.write("")


elif selected == "DataViz":
    st.write("<h2 style=text-align:center;>Data visualisation</h2>", unsafe_allow_html=True)
    choix = ['Géographie du trafic',
             'Top 20 compteurs (moyenne horraire)',
             'Evolution du nombre de passage par jour', 
             'Comparaison du flux en fonction du jour en semaine ou week-end', 
             'Répartition en fonction des jours de la semaine',
             'Nombre de passage de vélo par mois',
             'Affluence en fonction de chaque heure de la journée (1 courbe par mois)'] 
             
            
    option = st.selectbox('Choix de la DataViz', choix)
    df = pd.read_csv('df_viz.csv', sep=',')
    if option == 'Evolution du nombre de passage par jour':
        st.write("Vous avez choisi : Evolution du nombre de passage par jour")
        st.image('Viz_00.png')
        st.write("Ici nous pouvons constater une évolution non linéaire mais avec une constante régulière sur cinq jours avec plusieurs jours (en semaine) avec un fort passage de vélos sur les compteurs, et un à deux ou les valeurs sont basses (le weekend).")
    
    elif option == 'Comparaison du flux en fonction du jour en semaine ou week-end':
        st.write("Vous avez choisi : Comparaison du flux en fonction du jour en semaine ou week-end")
        st.image('Viz_01.png')
        st.write("Sur ce graphique nous voyons clairement ce que nous pouvions penser par rapport au plot présentant le nombre de passages par jour. Nous remarquons une moyenne de 80 cyclistes par heure qui passe alors qu’en weekend nous somme à 55 cyclistes par heure passant devant un compteur")
        #fig = plt.figure()
        #sns.catplot(x="Weekend", y='Count', kind='bar', data=df)
        #plt.title("Semaine/Weekend")
        #plt.xticks()
        #st.pyplot(fig)
        
    elif option == 'Répartition en fonction des jours de la semaine':
        st.write("Vous avez choisi : Répartition en fonction des jours de la semaine")
        st.image('Viz_02.png')
        st.write("Les pourcentage de répartition du nombre de cycliste passant sur un compteur en fonction du jour montre bien un passage plus important sur les trois jours ouvrés au centre d’une semaine de travail type (mardi, mercredi, jeudi sont respectivement à 16%). Le lundi et le mardi sont quant à eux respectivement à 14% (une explication probable est l’utilisation plus forte du télétravail sur ces deux jours).")
        st.write("Le samedi est quant à lui à 12 % et le dimanche qui est un jour où une grande majorité des activités sont inaccessibles est à 9% qui peut montrer que le vélo est moins utilisé lorsque nous n’avons pas besoin d’aller au travail ou faire des petites courses à proximité. ")
    
    elif option == 'Nombre de passage de vélo par mois':
        st.write("Vous avez choisi : Nombre de passage de vélo par mois")
        st.image('Viz_03.png')
        st.write("Ce graphique montre la différence sur les mois en hiver (environ 4 Millions par mois de novembre à février). Cependant en août, nous voyons une baisse du nombre de comptages par rapport aux mois qui l’entourent, une explication possible est les congés pris sur le mois d’août par les personnes habitant sur Paris ce qui explique le fait qu’il y ait moins de passage pendant cette période.")

    elif option == 'Affluence en fonction de chaque heure de la journée (1 courbe par mois)':
        st.write("Vous avez choisi : Affluence en fonction de chaque heure de la journée (1 courbe par mois)")
        st.image('Viz_04.png')
        st.write("Maintenant que nous avons regardé par période de la journée, nous pouvons regarder l’évolution heure par heure, Nous constatons donc deux pics (le matin et l’après-midi) avec entre les deux un nombre de passage par heure stable entre 10 heures et 15 heures.")
        st.write("En dépliant le graphique précédent, nous pouvons faire une comparaison de l’affluence par heure pour chaque mois. Ici nous pouvons remarquer que qu’importe le mois analyser nous retrouvons la même tendance sur tous les mois analysés (deux pics avec une stabilité entre ces deux pics).")

    elif option == 'Top 20 compteurs (moyenne horraire)':
        st.write("Vous avez choisi : Top 20 compteurs (moyenne horraire)")
        st.image('Viz_05.png')
        st.write("En prenant le nombre moyen de passages pris en compte par les compteurs par heure, cela nous permet de faire une comparaison plus significative entre tous les compteurs en prenant en compte la date d’installation de chaque compteur qui peut différer et de prendre en compte également le nombre de fois où le compteur s’active.")  
        st.write("Sur ce graphique nous voyons donc un compteur (73 boulevard de Sébastopol) qui est plus emprunté que les autres (50 passages de plus enregistrés par heure), les autres compteurs sur le Top 20 sont entre 100 et 200 passages par heure.")

    elif option == 'Géographie du trafic':
        st.write("Vous avez choisi : Graphique par arrondissement")
        st.write("")
        st.write("<h4>Situation géographique des compteurs</h4>", unsafe_allow_html=True)
        st.write("")
        m = folium.Map(location = [48.856578, 2.351828], zoom_start = 12, min_zoom=10, max_zoom=15)
        Paris_Dists = gpd.read_file('arrondissements.geojson')
        Dist_mean = df.groupby(["c_ar"], as_index = False)['Count'].mean()

        Dist_mean = pd.DataFrame({"Dist" : Dist_mean["c_ar"],
                                  "Count" : Dist_mean["Count"]})
        
        m.choropleth(geo_data = Paris_Dists,
                         key_on = "feature.properties.c_ar",
                         data = Dist_mean,
                         columns = ["Dist", "Count"],
                         fill_color = "YlOrBr",
                         legend_name = "Comptage horraire par arrondissement")
        
        df_adresse = df.groupby(['Adresse','Longitude','Latitude'], as_index = False)['Count'].mean()
        for index, row in df_adresse.iterrows():
            size = row['Count']/10  # Taille du point proportionnelle au comptage
            folium.CircleMarker([row['Longitude'], row['Latitude']], radius=size, fill=True).add_to(m)
            folium.Marker(location=[row['Longitude'],row['Latitude']], popup=row['Adresse']).add_to(m)
        folium_static(m, width=1000, height=600)
        st.write("")
        st.write("Sur la cartographie de compteurs, nous pouvons également visualiser, de manière très précise, et comparer l’intensité de trafic cycliste par compteur. Nous avons constaté que les compteurs avec le comptage horaire le plus important sont situés dans les arrondissements du centre-est de Paris. Cette visualisation de précision nous permet de confirmer l’observation plus haute selon laquelle le flux de vélo est concentré vers le centre-est de Paris. ")     
        st.write("")
        st.write("")
        st.write("<h4>Comptage horraire par arrondissement</h4>", unsafe_allow_html=True)
        st.write("")
        st.image('Viz_06.png')
        st.write("Nous avons constaté qu’il n’y a pas de données dans le 6ème et le 9ème arrondissement, car il n’y a pas de compteur installé dans ces périmètres, comme nous pouvons voir sur la cartographie de compteur")  
        st.write("Nous avons constaté également que le flux de cyclistes est plus important dans le centre de Paris, notamment entre le 2nd et le 11ème arrondissement. Nous avons pu également constater que le flux de vélo est plus important vers l’est que vers l’ouest.")
                
elif selected == "Modélisation":
    st.write("<h2 style=text-align:center;>Modélisations</h2>", unsafe_allow_html=True)
    st.write("")
    st.image('Processus de ML.jpg')
    st.write("<p style=text-align:center;>Les étapes de modélisation</p>", unsafe_allow_html=True)
    st.write("")
    st.write("Après le pré-traitement de données, et avant le commencement du Machine Learning, nous avons réalisé des Analyses ANOVA complété par f-régression sur nos variables pour nous permettre d’enlever les variables explicatives qui n’avaient aucune pertinence d’utilisation car elles auraient été un risque de biaiser notre résultat.")
    st.write("Pour rappel, dans notre projet, le but du Machine Learning consiste à prédire les valeurs de comptage de vélo pour chaque compteur (variable quantitative). Notre problématique est donc de l’ordre de régression. Pour répondre à notre problématique de machine Learning, nous passons par un traitement de type régression car il permet d’approcher une variable à partir d’autres qui lui sont corrélées.")
    st.write("En somme, nous avons donc entraîné trois modèles de régression non-linéaire (Decision Tree, Random Forest Regression et KNN), trois modèles avancés (AdaBoost Regression, XGBoost Regression, Gradient Boosting Regression) ainsi que quatre modèles de régression linéaire (Linear Regression, Ridge, Lasso, ElasticNetCV).")
    st.write("")
    st.image('models.jpg')
    st.write("<p style=text-align:center;>Les modèles de régression</p>", unsafe_allow_html=True)
    st.write("")
    st.write("Pour tester les modèles de régression, nous avons commencé par entraîner les modèles linéaires, tels que la régression linéaire simple, Ridge, Lasso et Elastic NetCV. Les performances sont très médiocres. Les scores obtenus sont autour de 0,31.")
    st.write("Nous procédions ensuite à tester les modèles non-linéaires tels que Decision Tree, Random Forest et KNN, les K plus proches voisins. Les scores sont nettement meilleurs. Par exemple, pour Random Forest, nous avons un score train de 0,92 et score test de 0,83, comme vous pouvez observer sur le tableau des résultats.")
    st.write("Nous passons donc aux modèles boost, plus rigoureux. Parmi les 3 modèles boost testés, le modèle AdaBoost basé sur Decision Tree nous a permis d’obtenir des résultats un peu meilleurs que le modèle Decision Tree simple. Il a boosté le score test de 2 points, ce qui a permis de corriger légèrement le surajustement de Decision Tree. Ses scores ne sont cependant pas meilleurs que Random Forest.")
    st.write("")
    st.write("")


    st.write("<h3 style=text-align:center;>Consultez les résultats de nos modèles entraînés et les comparaisons</h3>", unsafe_allow_html=True)
    st.write("")
    choix_results = ["Résultats de l'ensemble des modèles entraînés",
                     "Résultats des meilleurs modèles",
                    'Comparaison Decision Tree et Random Forest: Prédiction',
                    'Comparaison Decision Tree et Random Forest: Feature importances']
    
    option_results = st.selectbox('Votre choix', choix_results)
    
    if option_results =="Résultats de l'ensemble des modèles entraînés":
        st.write("<h3 style=text-align:center;>Résultats de nos modèles entraînés", unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.image('tableau_resultats.png')
        st.write("")

    if option_results =="Résultats des meilleurs modèles":
        st.write("<h3 style=text-align:center;>Résultats des meilleurs modèles", unsafe_allow_html=True)
        st.write("Après l’analyse des résultats, nous avons fait le choix de nous focaliser sur les modèles Decision Tree et Random Forest Regression. Cependant les deux modèles présentent un effet de surajustement assez marqué. Même si les scores des deux modèles retenus soient très proches, l’écart entre les score d'entraînement et de test se retrouve être plus faible pour Random Forest Regression ainsi que son RMSE plus petit, nous pensons donc qu’il est légèrement supérieur à l’autre avec un effet de surajustement moins marquant.")
        st.write("")
        st.image('bests.png')
        st.write("")

    if option_results =='Comparaison Decision Tree et Random Forest: Prédiction':
        st.write("<h3 style=text-align:center;>Comparaison des prédictions Decision Tree et Random Forest</h3>", unsafe_allow_html=True)
        st.write("Dans la comparaison de ces deux graphiques on peut clairement constater que les deux modèles sont extrêmement proches dans leur résultat prédictif. La proximité des deux modèles se ressentira également dans les features importances que nous retrouvons ci-dessous.")

# Display images side by side using st.columns()        
        image1 = "DTR_pred.png"
        image2 = "RFR_pred.png"

        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption='Decision Tree', use_column_width=True)
        with col2:
            st.image(image2, caption='Random Forest', use_column_width=True)

    if option_results =='Comparaison Decision Tree et Random Forest: Feature importances':
        st.write("<h3 style=text-align:center;>Comparaison des variables entre Decision Tree et Random Forest</h3>", unsafe_allow_html=True)
        st.write("")

# Display images side by side using st.columns()        
        image3 = "DTR_features.png"
        image4 = "RFR_features.png"

        col1, col2 = st.columns(2)
        with col1:
            st.image(image3, caption='Decision Tree', use_column_width=True)
        with col2:
            st.image(image4, caption='Random Forest', use_column_width=True)

    st.write("")
    


elif selected == "Entraînez votre modèle : À vous de jouer !":
    t0 = time()
    st.write("")
    st.write("<h2 style=text-align:center;>À vous de jouer!!</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("<h3 style=text-align:center;>Choisissez votre modèle à entraîner et comparez les résultats ! </h3>", unsafe_allow_html=True)
    st.write("Note: Pour cette démonstration, nous utilisons des données allégées pour entraîner les modèles. Les scores seront moins bons que notre entraînement avec des données plus complètes.")

    st.write("")
        
    
    # Chargement des données

    X_train = pd.read_csv("X_train_s.csv")
    X_test = pd.read_csv("X_test_s.csv")
    y_train = pd.read_csv("y_train_s.csv")
    y_test = pd.read_csv("y_test_s.csv")
    

    # Chargement du modèle
    #DecisionTreeRegression = joblib.load('DTR.joblib') 
    #AdaBoostRegression = joblib.load('DTR_ABR.joblib') 
    #RandomForestRegression = joblib.load('RFR.joblib')
    #XGBoostRegression = joblib.load('XGBR.joblib')
    #KNN = joblib.load('KNN.joblib')
    #RFR_ABR = load('RFR_ABR.joblib')
    #Modélisation
    def prediction(regressor):
        if regressor == 'Decision Tree Regression':
            clf = DecisionTreeRegressor(random_state=42)
            clf.fit(X_train, y_train)
            
        elif regressor == 'Random Forest Regression':
            clf = RandomForestRegressor(random_state=42)
            #clf = RandomForestRegressor
            clf.fit(X_train, y_train)
        
        elif regressor == 'AdaBoost Regression + Decision Tree':
            DTR = DecisionTreeRegressor(random_state=42)
            clf = AdaBoostRegressor(estimator=DTR, learning_rate=0.1, n_estimators=100)
            clf.fit(X_train, y_train)
         
        elif regressor == 'XGBoost Regression':
            clf = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            clf.fit(X_train, y_train)
        
        elif regressor == 'KNN - K-nearest neighbors':
            clf = KNeighborsRegressor(n_neighbors=3, metric='minkowski')
            clf.fit(X_train, y_train)
        
        elif regressor == 'Gradient Boosting Regression':
            #DTR = DecisionTreeRegressor(random_state=42)
            clf = GradientBoostingRegressor(random_state=42)
            clf.fit(X_train, y_train)
        
                
        return clf
    
    def scores(clf, choice):
        if choice == 'Scores':
            mae_train = mean_absolute_error(y_train, clf.predict(X_train))
            mse_train = mean_squared_error(y_train, clf.predict(X_train),squared=True)
            rmse_train = mean_squared_error(y_train, clf.predict(X_train),squared=False)
            mae_test = mean_absolute_error(y_test, clf.predict(X_test))
            mse_test = mean_squared_error(y_test, clf.predict(X_test),squared=True)
            rmse_test = mean_squared_error(y_test, clf.predict(X_test),squared=False)
            score_train = clf.score(X_train, y_train)
            score_test = clf.score(X_test, y_test)

            data = {'Score train': score_train,
                     'Score test': score_test,
                     'MAE train': mae_train,
                     'MAE test': mae_test,
                     'MSE train': mse_train,
                     'MSE test': mse_test,
                     'RMSE train': rmse_train,
                     'RMSE test': rmse_test}
            
            st.dataframe(data)
        
            #return score_train, score_test, mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test, 
            

        elif choice == 'Comparaison des valeurs réelles et de la prédiction':
            df_pred = pd.DataFrame({'Valeurs réelles': y_test.values.flatten(), 'Prediction': clf.predict(X_test).flatten()}).head(20)
            st.dataframe(df_pred.head(20), hide_index=True)
            #return df_pred
        
        elif choice == 'Graphique : les valeurs réelles vs la prédiction':
            pred = pd.DataFrame({'Valeurs réelles': y_test.values.flatten(), 'Prediction': clf.predict(X_test).flatten()})
            fig = plt.figure(figsize = (10,10))
            plt.scatter(data = pred, x = 'Prediction', y = 'Valeurs réelles', c='pink')

            plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
            plt.xlabel("prediction")
            plt.ylabel("valeurs réelles")
            plt.title('Prédiction du modèle vs valeurs réelles')
            st.pyplot(fig)
   
    choix = ['Decision Tree Regression',
             'Random Forest Regression',  
             'KNN - K-nearest neighbors',
             'AdaBoost Regression + Decision Tree',
             'XGBoost Regression',
             'Gradient Boosting Regression']
    
    option = st.selectbox('Choix du modèle à entraîner', choix)
    st.write('Le modèle choisi est :', option)

    clf = prediction(option)
    display = st.radio('Que souhaitez-vous montrer ?', ('Scores', 'Comparaison des valeurs réelles et de la prédiction', 'Graphique : les valeurs réelles vs la prédiction'))
    result = scores(clf, display)
    result
    #st.write(f"{display}: {result}")
    t1 = time()-t0
    st.write("Réalisé en {} secondes".format(round(t1,3)))
    


elif selected == "Interprétabilité":
    st.write("")
    st.write("<h2 style=text-align:center;>Interprétabilité avec SHAP</h2>", unsafe_allow_html=True)
    st.write(" ")
    st.write("Pour évaluer l’importance de chaque variable explicative ainsi que son influence sur notre variable cible, nous avons recours aux moyens de f-régression, de la librairie SHAP et aussi l’attribut feature_importances_ des modèles d'entraînement. Nous avons utilisé f-régression avec la corrélation et l’analyse ANOVA avant la modélisation pour avoir une vision de l’ensemble de nos variables. Cependant, f-régression, basée sur l’hypothèse de linéarité qui mesure la relation linéaire entre deux variables n’est pas une méthode adéquate pour analyser nos variables car nos données sont non-linéaires. De ce fait, nous nous appuyons sur l’attribut feature_importances_ des modèles d'entraînement et les méthodes de la librairie SHAP pour l’analyse de l’interprétabilité de nos variables.")
    st.write("Nous avons sélectionné un tree_based explainer pour effectuer l’analyse SHAP car nos données sont non-linéaires. Nous avons entraîné SHAP sur la base du modèle de Decision Tree Régression qui a donné des résultats satisfaisants avec l’avantage d’être plus léger que le modèle Random Forest.")

    st.write("Compte tenu du volume de nos données (779 144 entrées, il faudrait plusieurs jours pour entraîner un modèle SHAP pour obtenir les valeurs SHAP), nous procédons d'abord à alléger les données.")
    st.write("")
    st.write("Nous prélevons les échantillons avec deux méthodes en guise de comparaison :")
    st.write("➢	Méthode n°1 : ")
    st.write("○	Prélever les données de nos données d'origine (données de df) à partir de la 6 e semaine de l’année 2022 (à partir de la 2ème semaine du février) puis une semaine de données à l’espace de toutes les 5 semaines. Nous souhaitons alléger davantage nos données en faisant le possible pour garder une semaine de données pour chaque mois. Entre 4 et 5 semaines, nous avons choisi 5 pour alléger davantage, il est donc possible de manquer les données d’un mois. Il s’agit d’un compromis nécessaire entre la représentabilité d’échantillonnage et l’allégement de données.")
    st.write("")
    st.write("➢	Méthode n°2 : ")
    st.write("○	Prélever 10 000 échantillons de manière aléatoire des données d'X_train avec la méthode d'échantillonnage de SHAP.")
    st.write("Finalement, les interprétations sont assez similaires avec les deux méthodes d'échantillonnage, avec une légère variation. Nous supposons cependant que le 1er échantillonnage qui respecte l’ordre chronologique de nos données devrait représenter mieux l'ensemble de nos données d'entraînement. Comme nous l'avons expliqué plus haut, l’échantillonnage aléatoire semble avoir l'incidence sur l’entraînement et la prédiction de nos données.")
    st.write("")
    
    image5 = "SHAP_light.png"
    image6 = "SHAP_X10000.png"

    col1, col2 = st.columns(2)
    with col1:
        st.image(image5, caption='Résultat avec 1er échantillonnage', use_column_width=True)
    with col2:
        st.image(image6, caption='Résultat avec 2ème échantillonnage (aléatoire)', use_column_width=True)

    st.write("")
    st.write("<h2 style=text-align:center;>Interprétation des variables</h2>", unsafe_allow_html=True)
    st.write("D'après les analyses SHAP, nous avons quelques remarques:")
    st.write("En premier lieu, nous avons pu constater que les variables temporelles (heure, jours de la semaine, mois) et spatiales (géographiques, telles que les arrondissements, latitudes et longitudes) sont les variables les plus importantes pour nos prédictions, par rapport à d’autres types de variables tels que les évènements (grèves etc.), la météo).")
    st.write("En second lieu, parmi les variables temporelles, les variables liées à l’horaire et la période de la journée sont plus importantes que celles liées aux jours de la semaine ou au mois. Bien que nos données semblent, en outre, manifester une certaine « saisonnalité » au rythme des saisons, il reste difficile pour l’algorithme d’extraire cette saisonnalité sur 13 mois de données.")
    st.write("Pour finir, parmi les variables à caractère spatial/géographique, la latitude et l’arrondissement sont particulièrement importantes. Cela confirme l'observation dans notre data visualisation de la répartition des valeurs de comptage par arrondissement. D’après les plots que vous retrouverez sur la page suivante, le trafic de vélo est plus important au centre de Paris par rapport au pourtour, plutôt vers la rive droite (nord) que la rive gauche (sud), et plus vers l’est que l’ouest. Cette dépendance spatiale et géographie de la circulation de vélo est donc confirmée.")
    st.write("")
    st.image('Viz_07.png')
    st.write("")
    st.image('Viz_08.png')
    st.write("")
    st.write("<h5 style=text-align:center;>Géographie du comptage selon latitude et longitude</h5>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("<h2 style=text-align:center;>Conclusion</h2>", unsafe_allow_html=True)
    st.write("Le fait que les valeurs de comptage dépendent des variables journalières semble suggérer que le comptage de vélo est très lié aux activités quotidiennes des Parisien(nes), au rythme du travail, de l’étude, et probablement de loisir.")
    st.write("Ainsi, cette dépendance spatio-temporelle nécessite des analyses avancées, et prendre en compte des données de la population (âge, démographie, revenu, profession, niveau d’études/diplômes, origine, opinion politique, etc.), des activités (économiques, culturelles, de loisir, écoles et établissements de l’enseignement supérieur par exemples) dans les différents arrondissements. Ces données ne sont pas incluses dans notre étude mais notre travail et les analyses de l’influence des variables nous ouvrent la voie vers des études futures, plus approfondies.")
    st.write("")


elif selected == "Bilan et perspective":
    st.write("")
    st.write("<h2 style=text-align:center;>Bilan</h2>", unsafe_allow_html=True)
    
    st.write("L'objectif étant de comprendre l'évolution du trafic cycliste en fonction des secteurs de la ville et des périodes clés, ainsi que les facteurs qui influencent le choix entre le vélo, la voiture et les transports en commun : ")
    st.write("  ➢	Nous constatons que le trafic est plus élevé la semaine que le week-end, cela s’expliquant par l’utilisation du vélo pour les déplacements au travail.")
    st.write("  ➢	Depuis la pénurie du carburant de novembre 2022 et l’augmentation des prix à cette période peut expliquer une utilisation du vélo plus forte.")
    st.write("  ➢	Nous remarquons que le nombre de vélo est en baisse pendant le mois d'août, cela étant certainement dû aux départs en vacances durant cette période")
    st.write("")
    st.write("")
    st.write("<h2 style=text-align:center;>Pistes d’amélioration</h2>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("amelioration.jpg")

    with col3:
        st.write("")
    
    #st.image('amelioration.jpg')
    st.write("")
    st.write("  ⮚	Utiliser les informations spatiales et temporelle en même temps pour améliorer la performance :")
    st.write("      o	Etant donnée que nos données sont composées des informations spatiales (coordonnées géographiques, adresse, etc.) et temporelles (heure, date etc.), il sera intéressant d’utiliser les modèles qui capturent les relations spatiales et temporelles à la fois. Nous espérons pouvoir approfondir ce sujet et améliorer notre modèle dans ce sens.")
    st.write("      o	Nos données se caractérisent par une multiple périodicité : heure, jours de la semaine, mois, saison etc. Pour capturer la saisonnalité de nos données, l’idéal est d’avoir au moins 2 cycles de 4 saisons ou 2 cycles de 12 mois. Collecter les données des mois à venir en 2023 jusqu’au janvier 2024 pourra contribuer à améliorer la capacité de notre modèle à prédire.")
    st.write("  ⮚   Utiliser pipeline pour entraîner nos modèles.")
