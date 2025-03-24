# Projet Capstone

Ce projet a été développé en utilisant **Python** comme langage principal. Il s'agit d'une application interactive construite avec **Streamlit**, une bibliothèque Python permettant de créer des interfaces utilisateur pour des projets de science des données et d'analyse.

## Technologies utilisées

1. **Python** : Langage principal utilisé pour le développement.
2. **Streamlit** : Framework pour créer des applications web interactives.
3. **Matplotlib** : Bibliothèque pour la visualisation de données.
4. **Pandas** : Utilisé pour la manipulation et l'analyse des données.
5. **SQLAlchemy** : Utilisé pour la gestion des connexions à la base de données.
6. **Streamlit-Option-Menu** : Bibliothèque pour créer des menus de navigation interactifs.
7. **HTML/CSS** : Utilisé pour personnaliser le style et l'apparence de l'application.

## Fonctionnalités principales

- **Authentification** : Gestion des connexions utilisateur avec des rôles spécifiques (admin, utilisateur, etc.).
- **Tableaux de bord** : Visualisation des données et analyses interactives.
- **Prédictions** : Modèles de prédiction pour la consommation de données.
- **Gestion des comptes** : Création, modification et affichage des comptes utilisateurs.
- **Base de données** : Connexion et gestion des données via une base de données relationnelle.

## Prérequis

- Python 3.8 ou supérieur
- Les bibliothèques suivantes doivent être installées :
  ```bash
  pip install streamlit pandas matplotlib sqlalchemy streamlit-option-menu
  ```

## Lancer le projet

Pour exécuter l'application Streamlit, suivez les étapes ci-dessous :

1. **Assurez-vous que toutes les dépendances sont installées** :
   Installez les bibliothèques nécessaires en exécutant la commande suivante dans le terminal :
   ```bash
   pip install -r requirements.txt
   ```
   Si vous n'avez pas de fichier `requirements.txt`, utilisez la commande mentionnée dans la section "Prérequis".


3. **Lancez l'application Streamlit** :
   Exécutez la commande suivante pour démarrer l'application :
   ```bash
   streamlit run main.py
   ```

4. **Ouvrez l'application dans votre navigateur** :
   Une fois la commande exécutée, un lien sera affiché dans le terminal (par exemple, `http://localhost:8501`). Cliquez sur ce lien ou copiez-le dans votre navigateur pour accéder à l'application.

## Auteur

Ce projet a été développé dans le cadre d'un stage de fin d'études.