-------------------------------------------------
--- Projet OCR: Optical Character Recognition ---
-------------------------------------------------

--- Participants du projet:

	- Benjamin Tabet
	- Samy Mazouz
	- Ari-Vinoth Sekar


--- Description des fichiers:

	1/ Data_Vizualisation.ipynb:
		- NoteBook présentant le jeu de données utilisé pour l'entrainement du modèle de DeepLearning

	2/ Extractor tgz .ipynb:
		- NoteBook permettant l'extraction de fichier tgz dans un répertoire précisé en paramêtre

	3/ txt_to_csv.ipynb:
		- NoteBook permettant la transformation d'un fichier 'txt' précisé en paramêtre en fichier 'csv'

	4/ Boudingbox_IOU.ipynb:
		- Analyse des box de nos formulaires avec Tessereact accompagné de la métrique IOU

	5/ DeepLearning_Detection_model.ipynb:
		- NoteBook de mise en place et entrainement du model de Deep Learning permettant la lecture et prédiction des mots sur une image

	6/ Demonstration.ipynb:
		- NoteBook de demonstration: Nous prenons en entrée un formulaire sélectionné aléatoirement.
		- Boxing des mots à l'aide de Tessereact
		- Prédiction des mots à l'aide du model de deepLearning préentrainé
		- Affichage du formulaire, composé des mots encadrés accompagné des mots prédits.

	7/ Répertoire utils composé des fichiers suivant:
		- functions.py
			fonctions utilisé pour l'entrainement du model de deepLearning et l'encadrement des mots lors de la démonstration
		- SamplePreprocessor.py


--- Fichiers requis pour le démarrage des NoteBook:

	- Répertoire words, télécharger sur le net composé de toutes les images des mots ayant servi à l'entrainement du model

    - Répertoire utils incluant les fichier python de fonctions.

    - Répertoire model_08112021_allData_final, composé du model pré-entrainé

	- Fichier words.csv indiquant les chemins de chaque images 'words'

	- Fichier forms.csv indiquant les chemins de tous les images des formulaires. 
