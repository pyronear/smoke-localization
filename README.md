# Smoke localization

## Lidar data
Go to [`./lidar`](./lidar)  
Install [`requirements.txt`](./lidar/requirements.txt) in your environment:  
`pip install requirements.txt`  

### Download the data
- either from [IGN geoservice](https://geoservices.ign.fr/lidarhd) where you can select the tiles you want
- or directly from the notebook [`visualize_terrain.ipynb`](./lidar/visualize_terrain.ipynb), where a preselection of tiles is available in [`liste_dalle.txt`](./lidar/data/MNS/liste_dalle.txt)  

### Explore
Open [`visualize_terrain.ipynb`](./lidar/visualize_terrain.ipynb) and follow the steps.


## cameratransform

`cameratransform` est un package Python simple de modélisation de caméras dans orientées et localisées dans l'espace.
Il peut être 
utilisé pour réaliser des projections de points depuis l'espace caméra vers l'espace monde et vice-versa. Il possède 
quelques fonctionnalités additionnelles, telles que des outils de ray tracing, évaluation des distances et des 
hauteurs d'objets, projection du champ de vue de la caméra...

La branche `exploration/cameratransform` contient un notebook `cameratransform/cameratransform_test.ipynb` de test/viz des capacités du package à projeter des 
coordonnées IGN sur le champ de vue d'une caméra (ici brison_4, cf. les [données et galeries d'images de caméras Pyronear](https://drive.google.com/file/d/1GsJIjNyjnZjV2tzMuB0xTZ2hwz-lpRjB/view?usp=sharing)). Les données IGN du département correspondant (Ardèche) au format ASCII grid résolution 25m sont téléchargeables [ici](https://wxs.ign.fr/aqd29otkz2hofiee5pb0fygn/telechargement/prepackage/BDALTI-25M_PACK_FXX_2023-02-01$BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16/file/BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16.7z) au format compressé 7zip.

Les étapes nécessaires à l'exécution du notebook sont donc les suivantes :
- Installation des packages nécessaires : ` pip install -r cameratransform/requirements.txt`
- Téléchargement des [données et galeries d'images de caméras](https://drive.google.com/file/d/1GsJIjNyjnZjV2tzMuB0xTZ2hwz-lpRjB/view?usp=sharing)
- Téléchargement et extraction des [données IGN utilisées](https://wxs.ign.fr/aqd29otkz2hofiee5pb0fygn/telechargement/prepackage/BDALTI-25M_PACK_FXX_2023-02-01$BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16/file/BDALTIV2_2-0_25M_ASC_LAMB93-IGN69_D007_2022-12-16.7z)
- Édition de la cellule du notebook qui indique les chemins de dossiers contenant les données
- Exécution séquentielle du notebook

Le notebook a été testé sur Jupyter Lab, dans un environnement virtuel généré à l'aide du fichier `requirements.txt` avec Python 3.10.12 sur Ubuntu 22.04.
