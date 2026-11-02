# Recherche supervisee FD001 avec Rustuna

Cette etape precede l'entrainement final. Aucun test officiel ni fichier FD002
n'est lu. La recherche selectionne uniquement sur la validation source.

## Installation et commandes

Python >=3.11 pour cette option (le pipeline sans HPO reste compatible >=3.10).
Depuis la racine du depot, dans l'environnement Python du projet :

```bash
python -m pip install -e '.[test]'
python -m pip install -r requirements-hpo.txt
python -m pytest -ra
cmapss smoke --output runs/smoke_before_hpo
python -m cmapss.hpo --config configs/hpo_fd001.yaml --output runs/hpo_fd001_s42
cmapss train --config runs/hpo_fd001_s42/best_config.yaml --output runs/source_fd001_tuned_s42
```

Ces commandes doivent etre executees reellement par l'agent local. Les repertoires
smoke/search/train doivent etre nouveaux. La recherche ne lance pas elle-meme
l'entrainement final. Ne pas lancer evaluate, transfer ou adapt a cette etape.

## Protocole

- TPE Rustuna, sequentiel, 20 essais tentes au total, reference comprise.
- Essai numero zero : configuration source de reference, meme budget court.
- Chaque essai : au maximum 30 epochs, patience 8, selection du checkpoint sur
  la RMSE de validation equilibree entre moteurs, en cycles.
- Parametres recherches : learning_rate log [1e-4,3e-3], weight_decay log
  [1e-6,1e-3], batch_size [64,128], hidden_size [32,64,128], num_layers [1,2],
  dropout [0,0.2,0.4]. Toutes ces bornes sont dans le YAML HPO.
- Fenetre=30, stride validation=10, plafond=125, target_scale=125, seed modele
  et split=42 restent ceux du YAML source et ne sont pas recherches. La recherche
  refuse une dimension supplementaire telle que rul_cap ou validation_fraction.
- Memes moteurs, memes points de fin des fenetres de validation et meme
  preprocessing source pour tous les essais. Les manifests split/preprocessing
  des checkpoints sont verifies, ainsi que l'empreinte du train source.
- Minimiser la meilleure metrique source de chaque essai, jamais un score test.
  La valeur est celle de best.pt, pas une metrique terminale officielle.
- Pas de pruning inter-essais ni d'AMP ajoute : early stopping supervise existant.
  Aucun fallback silencieux vers Optuna. Pas de parallelisme sur un GPU unique.

Le budget court favorise potentiellement les configurations convergeant vite.
Le gagnant n'est pas garanti optimal avec un budget long. Le minimum obtenu
sur cette validation repetee n'est pas une estimation non biaisee de generalisation.
Une seule seed sert a la recherche ; ce n'est pas une validation multiseed.
Rustuna accelere le moteur de recherche, pas les epochs PyTorch elles-memes.

## Artefacts et reprise

study.db conserve l'historique Rustuna SQLite. Chaque trial_XXXXX contient tous
les artefacts d'un run supervise, plus result.json. A la racine :
search_manifest.json, source_protocol.json, trials.csv, best_trial.json,
best_config.yaml et search_summary.json. La version Rustuna exacte est enregistree.
Le YAML gagnant restaure epochs/patience du YAML source (100/15 par defaut) ;
il sert a un NOUVEL entrainement de zero, pas a poursuivre le checkpoint court.
Le split, la seed et le preprocessing restent ceux du protocole source.

```bash
python -m cmapss.hpo --config configs/hpo_fd001.yaml --output runs/hpo_fd001_s42 --resume
```

Reprise uniquement entre essais, avec config, source, Python, version Rustuna et
repertoire identiques. n_trials est un total, pas un nombre d'essais additionnels.
Le sampler est reinitialise avant chaque essai avec search_seed+numero_essai,
en consultant l'historique SQLite : meme regle pour une execution continue ou
reprise a la frontiere d'un essai. Cette convention differe d'un TPE conserve
continuellement en memoire. Pas de promesse bit-a-bit entre versions/materiels.
Un etat RUNNING/WAITING apres interruption provoque un refus explicite : inspecter
l'etat au lieu de pretendre reprendre exactement un essai incomplet. Un echec
technique arrete la commande ; il n'est pas remplace par une fausse valeur finie.
Le CSV enregistre les etats COMPLETE/FAIL. Les temps du summary concernent le
segment courant, ceux de chaque run sont dans ses propres artefacts.

## Validation de l'ajout

Dans l'environnement de preparation, PyTorch et Rustuna sont absents et le
reseau Python est desactive. Les tests purs utilisent explicitement un double
de test pour verifier l'orchestration ; ils ne valident pas l'API Rustuna reelle.
Deux tests d'integration sont fournis : backend SQLite Rustuna et petit
entrainement Rustuna+PyTorch sans aucun fichier test officiel/FD002. Ils sont
ignores si leurs dependances sont absentes. Aucun essai HPO reel ni resultat
scientifique n'est annonce. L'agent local doit installer l'option et executer
la suite complete avant de lancer les 20 essais.

References API : documentation officielle Rustuna, Getting Started,
https://rustuna.readthedocs.io/en/latest/tutorial/getting-started/
