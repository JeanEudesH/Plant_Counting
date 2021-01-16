# Plant_Counting
 Code to automatically detect plants on images of crops fields captured by an UAV


 #### Problèmes d'import
 Sur mon PC, les imports ne fonctionnaient pas en utilisant le code initial:
 ```os.chdir("../Utility)
    import general_IO as gIO```
J'ai résolu le problème en ajoutant au PYTHON PATH (sys.path) le répertoire racine "Plant_Counting".
Ce répertoire n'est pas ajouté dynamiquement en lançant le script Whole_Process.py (ce doit être le 
répertoireee Whole_Process qui est ajouté).

Voir : http://sametmax.com/les-imports-en-python/
