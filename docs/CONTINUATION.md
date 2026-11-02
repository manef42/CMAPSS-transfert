# Continuation sans perdre les resultats de la partie 1

## Reference source figee

Lire source_fd001_baseline.json dans ce dossier. Le checkpoint a utiliser est
best.pt du run source tune, pas last.pt. Le champ checkpoint_git_blob_sha1 est
un identifiant de blob Git, pas un SHA256. Verifier sa correspondance locale :

```bash
git hash-object runs/mission_20261102_1800_source_fd001_tuned_s42/checkpoints/best.pt
```

Valeur attendue : 594f77997febf4a6eeb5029d451b82ec141ee6ea.
Le resultat de validation 12.30539 cycles n'est pas un resultat test officiel.

## Protection et prerequis

- Rester sur manef. Inspecter git status ; ne jamais reset --hard, clean,
  forcer un checkout/pull ou ecraser des changements locaux. Ne synchroniser
  qu'en avance rapide avec un workspace propre ; sinon s'arreter et signaler.
- Ne pas relancer HPO ni train supervise, et ne pas utiliser --resume sur un
  ancien run. Tous les protected_runs et les donnees originales sont en lecture seule.
- Avant les commandes, sauvegarder un manifeste SHA256 recursif des anciens
  runs, hors de ces dossiers ; reverifier les memes fichiers apres les operations.
  Le commit de reference protege les fichiers suivis, pas les fichiers non suivis :
  sauvegarder hors depot tout travail local non versionne avant synchronisation.
- Utiliser l'environnement existant Python >=3.11 avec Torch et Rustuna, puis
  executer python -m pytest -ra. Archiver stdout/stderr et le code retour dans
  un NOUVEAU dossier de logs. Exiger les tests Torch/HPO sans skip de dependances.
- Recharger le checkpoint de confiance en eval et verifier des predictions
  source finies. Arreter en cas d'erreur ; ne pas contourner un garde-fou de donnees.
- Verifier train_FD001.txt, test_FD001.txt, RUL_FD001.txt, train_FD002.txt,
  test_FD002.txt et RUL_FD002.txt. Pas de telechargement implicite.
- Les checkpoints source proviennent de Windows. En cas de migration d'OS, les
  cles de chemins des empreintes peuvent faire refuser DANN malgre des octets
  identiques. Ne pas editer le checkpoint ou desactiver la verification : signaler
  le probleme pour une correction explicite de portabilite.

## Ordre des experiences

Executer depuis la racine, une commande apres l'autre, uniquement si la precedente
reussit. Les repertoires ci-dessous doivent etre inexistants et seront crees par
la CLI. S'ils existent, choisir avant le depart un nouvel identifiant coherent ;
ne jamais supprimer/vider un ancien repertoire pour reutiliser son nom.

La configuration DANN tuned est fixee AVANT de lire les scores de test. Elle
reprend l'architecture source 128/1/0, la fenetre et la cible ; les parametres
adversariaux restent le budget predefini, pas un reglage sur le test cible.

```bash
cmapss evaluate --checkpoint runs/mission_20261102_1800_source_fd001_tuned_s42/checkpoints/best.pt --data-dir src --dataset FD001 --output runs/phase2_eval_fd001
cmapss transfer --checkpoint runs/mission_20261102_1800_source_fd001_tuned_s42/checkpoints/best.pt --data-dir src --dataset FD002 --output runs/phase2_direct_fd002
cmapss adapt --config configs/dann_fd001_to_fd002_tuned.yaml --source-checkpoint runs/mission_20261102_1800_source_fd001_tuned_s42/checkpoints/best.pt --output runs/phase2_dann
cmapss evaluate --checkpoint runs/phase2_dann/checkpoints/last.pt --data-dir src --dataset FD002 --output runs/phase2_adapted_fd002
```

Aucun ajustement du modele source ou des hyperparametres apres lecture du test.
DANN utilise seulement les donnees source supervisees et train cible sans labels.
Il n'est pas source-free. Garder les memes fichiers test FD002 pour les deux
methodes et comparer leurs empreintes. Utiliser le dernier checkpoint DANN au
budget fixe, jamais choisir une epoch ou une seed sur le test cible.

## Bilan a archiver dans un nouveau dossier

Logs pytest, commandes/codes retour, commit et environnement, chemins des runs,
metriques RMSE/MAE/biais/NASA pour FD001 source et les deux evaluations FD002,
convention de cible plafonnee a 125 cycles, comparaison sur les memes moteurs,
diagnostics source/domaine et dispersion cible. Ne pas confondre metriques de
validation et metriques terminales. Ne pas promettre une amelioration DANN.
Verifier que le manifeste des anciens runs est inchange. Aucune modification ni
publication Git supplementaire sans autorisation explicite.

## Verification de cette preparation

Configuration YAML comparee statiquement aux champs de compatibilite du source ;
aucune evaluation reelle, aucun rechargement Torch et aucune adaptation executes
pendant la preparation de ces fichiers. Les verifications dynamiques sont a
executer par l'agent local avant les experiences.
