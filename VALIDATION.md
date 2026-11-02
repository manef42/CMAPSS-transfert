# Bilan factuel de validation

## Inspection

Branche cible : manef de manef42/CMAPSS-transfert.
Point de depart inspecte : fa57a05b10f90f15c6d5e9a9c132003b7ae78d94.
Le depot contient des fichiers C-MAPSS FD001..FD004 et un PDF dans src/ ; aucun
code applicatif ni declaration de dependances ne preexistait. Ces fichiers
originaux ne sont pas inclus dans les modifications.

## Execute dans l'environnement de preparation

- Python 3.12.8 ; NumPy, pandas, PyYAML, matplotlib et pytest disponibles.
- PyTorch absent ; acces reseau de l'environnement Python desactive.
- `python -m pytest -q -ra` : 19 passed, 1 skipped (1.81 s au premier passage).
  Le skip concerne le module entier tests/test_torch.py, pas un seul cas
  d'entrainement. Les tests de gradients et d'integration n'ont pas ete executes.
- Analyse syntaxique AST des 12 fichiers Python : succes.
- CLI synthetic puis inspect test FD001 : succes, 30 lignes, 5 moteurs,
  association terminale verifiee. Aucun FD002 necessaire.
- Le test pytest des figures a effectivement genere cinq PNG synthetiques
  (courbes supervisees, DANN, nuage terminal, histogramme et erreur/RUL).
- CLI smoke complet tente : echec avant entrainement, code retour 1,
  `ModuleNotFoundError: No module named 'torch'`.

## Non execute / non valide

Aucun entrainement PyTorch, checkpoint entraine, passage DANN, reprise numerique
ni smoke complet reussi dans cet environnement. Aucune experience sur donnees
NASA reelles, aucun RMSE ou score scientifique rapporte. La presence des fichiers
reels sur GitHub ne signifie pas qu'ils ont ete traites localement.

La CI ajoutee execute pytest et le smoke complet sur CPU apres installation de
PyTorch, mais son resultat n'etait pas disponible pendant la preparation. Cette
livraison ne revendique donc pas encore le critere « tous les tests et smoke
end-to-end passent reellement ». Le code correspondant est implemente ; sa
validation dynamique PyTorch reste necessaire. Voir README pour les commandes.
