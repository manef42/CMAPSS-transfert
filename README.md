# C-MAPSS : LSTM supervise et adaptation LSTM-DANN

Pipeline PyTorch compact, sans notebook ni telechargement implicite. Les donnees
originales et le PDF deja presents dans `src/` ne sont jamais modifies. Le code
Python est uniquement dans `src/cmapss/`. Aucun autre modele n'est implemente.

## Installation

Python >=3.10. Depuis la racine du depot :

```bash
python -m venv .venv
source .venv/bin/activate
# CPU ; pour CUDA, installer la distribution PyTorch adaptee a votre machine.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m pytest -ra
```

Les tests et les commandes n'utilisent pas le reseau une fois les dependances
installees. La CI installe les dependances explicitement. Les versions exactes
effectivement utilisees sont exportees dans chaque run ; ce projet n'impose pas
un lockfile universel CPU/CUDA. Reutiliser les memes versions pour une reprise.

## Donnees et inspection

Par defaut, les fichiers originaux restent dans `src/` :
`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`. FD002 est facultatif pour le
supervise. Adapter `data_dir` dans une copie du YAML pour un autre emplacement.

```bash
cmapss inspect --data-dir src --dataset FD001 --partition train
cmapss inspect --data-dir src --dataset FD001 --partition test
```

26 colonnes numeriques finies sans en-tete, espaces variables. IDs/cycles entiers
positifs, lignes triees, cycles continus commencant a 1. Toute anomalie est une
erreur explicite : pas de tri ni de correction silencieuse. L'association RUL
officielle est ligne i -> moteur i (IDs 1..N verifies), jamais ligne de capteurs.
Un ID dans un fichier n'identifie pas le meme moteur dans un autre fichier.

## Smoke test synthetique FD001 seul

```bash
cmapss smoke --output runs/smoke
```

Genere cinq moteurs locaux, entraine deux epochs sur CPU, selectionne sur les
moteurs de validation, recharge le checkpoint et evalue une prediction terminale
par moteur. Aucun fichier FD002 n'est cree ou requis. Les resultats synthetiques
ne constituent pas une mesure de performance scientifique.

## Entrainement supervise FD001

```bash
cmapss train --config configs/source_fd001.yaml --output runs/source_s42
cmapss evaluate --checkpoint runs/source_s42/checkpoints/best.pt --data-dir src --dataset FD001 --output runs/eval_fd001_s42
```

Le train ne lit que `train_FD001.txt`. L'evaluation officielle est une commande
separee et ne participe jamais a la selection. FD001 seul est un LSTM supervise,
pas une adaptation DANN. La tete de domaine n'est pas executee dans ce mode.

## Transfert direct FD001 -> FD002

```bash
cmapss transfer --checkpoint runs/source_s42/checkpoints/best.pt --data-dir src --dataset FD002 --output runs/direct_fd002_s42
```

Seuls `test_FD002.txt` et `RUL_FD002.txt` sont necessaires. Pas d'optimiseur, aucune
mise a jour de poids, aucun refitting du preprocessing ni reglage sur le test.
L'evaluation emploie exactement la fenetre, les variables et la cible du checkpoint.

## Adaptation non supervisee LSTM-DANN

```bash
cmapss adapt --config configs/dann_fd001_to_fd002.yaml --source-checkpoint runs/source_s42/checkpoints/best.pt --output runs/dann_s42
cmapss evaluate --checkpoint runs/dann_s42/checkpoints/last.pt --data-dir src --dataset FD002 --output runs/adapted_fd002_s42
```

Utilise le split source enregistre et les moteurs autorises de `train_FD002.txt`.
`target_units: null` autorise tous les moteurs du train cible ; une liste explicite
restreint cet ensemble et est sauvegardee au manifeste. Aucun fichier RUL cible,
aucun test cible et aucun label RUL reconstruit ne sont accessibles au Dataset
non labelle. Le train_step recoit seulement `target_x`, pas de labels cibles.
Cette adaptation reutilise le source : elle n'est pas source-free.

Le budget `adaptation_epochs` est fixe avant l'experience. Le dernier checkpoint
est le resultat principal : aucun `best.pt` cible. La validation source reste un
diagnostic, sans arret anticipe en DANN. Utiliser exactement les memes fichiers
test pour comparer transfert direct et adaptation (SHA256 exportes).

## Figures et plusieurs seeds

```bash
cmapss figures --run runs/source_s42
cmapss figures --run runs/eval_fd001_s42
cmapss figures --run runs/dann_s42
cmapss seeds --config configs/source_fd001.yaml --seeds 11 22 33 --output runs/source_multiseed
```

`figures` regenere explicitement les PNG a partir des CSV. Les autres commandes
refusent un repertoire existant sauf reprise explicite. La commande `seeds`
repete le protocole supervise, avec un split moteur reproductible par seed ; elle
ne choisit pas la meilleure seed sur le test. Pour comparer les methodes, adapter
chaque checkpoint source avec la meme seed (copie du YAML DANN avec `seed` adapte),
puis evaluer les deux methodes sur les memes tests. Aucune aggregation de metriques
sur des fenetres et des predictions terminales melangees.

## Choix scientifiques

### Cible et metriques

Train : RUL brute = dernier cycle du moteur - cycle courant. Test historique :
RUL brute = cycle terminal observe - cycle courant + RUL terminale officielle.
Ce dernier calcul existe seulement dans l'evaluation. La cible d'evaluation est
`min(RUL brute, rul_cap)` si le plafond est non nul. `rul_cap: null` le desactive.
`target_scale` est independant : la cible numerique d'apprentissage est la cible
transformee / target_scale. Par defaut plafond=125 et scale=125. La sortie
lineaire est remultipliee par target_scale pour toutes les metriques et figures.

RMSE, MAE, biais et score NASA sont calcules en cycles, sans ecretage des
predictions negatives. L'erreur e est prediction - cible d'evaluation :
score par moteur = exp(-e/13)-1 pour e<0, exp(e/10)-1 sinon. `nasa_sum` est la
somme selon cette formule ; `nasa_mean` est distinct. Avec le plafond active,
les scores comparent la prediction a la cible plafonnee, pas a la RUL brute :
ne pas les comparer sans precaution a un protocole sans plafond. Le CSV conserve
les deux verites pour lever l'ambiguite. Float64 pour les metriques ; un
depassement du score est signale par `nasa_overflow=true` et des valeurs JSON
null, jamais par un ecretage ou une fausse valeur finie.

### Splits et validation

Le split 80/20 est effectue avant les fenetres, exclusivement par moteur. La
validation utilise tous les points de fin de position 0, stride, 2*stride, ...
plus le dernier point (sans doublon). Ces fenetres et moteurs restent fixes.
Le padding est applique aux premiers points, donc la validation ne porte pas
seulement sur les dernieres cibles nulles des trajectoires completes.

Selection supervisee : pour chaque moteur i avec n_i fenetres, calculer
MSE_i = somme_j (prediction_ij - cible_ij)^2 / n_i ; le critere est
sqrt(somme_i MSE_i / nombre_de_moteurs). Chaque moteur a donc le meme poids,
independamment de sa longueur. Early stopping sur ce seul critere, avec
`patience` et `min_delta`. Le test officiel utilise une seule fenetre terminale
par moteur ; son RMSE ne mesure pas le meme echantillonnage de sante.

### Preprocessing et fenetres

24 variables conservees dans l'ordre : setting_1..3, s1..21. Moyenne/ecart-type
population ajustes uniquement sur les observations des moteurs source train,
avant fenetrage. Aucun scaler par trajectoire. Si std source < 0.001, centrer et
diviser par 1 unite brute ; sinon diviser par std source. Le seuil est
configurable et sauvegarde, sans suppression de variables ni amplification
arbitraire d'un ecart-type quasi nul. Cela ne garantit pas des amplitudes
similaires pour les nouvelles conditions : consulter les CSV de dispersion
(moyenne/std transformees, maximum absolu, fraction |x|>10). Les entrees non
finies ou hors capacite float32 produisent une erreur explicite.

Le preprocessing source reste fige en transfert et DANN. Les nouvelles conditions
FD002 peuvent rester tres hors distribution : ne pas refitter silencieusement
sur le test pour corriger cela. Toute autre normalisation serait un autre protocole.

Fenetre de longueur L finissant a t : cycles t-L+1..t et cible a t. Une fenetre
ne traverse jamais deux moteurs. Si moins de L observations sont disponibles,
repeter la premiere a gauche (train, validation et test). Le Dataset conserve
un tableau par moteur et un index (moteur, fin), pas une copie de chaque fenetre.
Par defaut les fenetres sont tirees avec remplacement avec poids 1/n_i : masse
totale identique par moteur, nombre de tirages egal au nombre de fenetres.
`sampling: windows` parcourt une permutation uniforme des fenetres et favorise
donc davantage les longs moteurs. La validation n'est jamais aleatoire.

### Architecture et objectif adversarial

LSTM unidirectionnel, batch_first, deux couches, hidden=64, dropout inter-couches
0.2. Representation = h_n[-1]. Tetes distinctes Linear(64,32), ReLU, Dropout,
Linear(32,1). Sortie RUL lineaire ; sortie domaine logit pour BCEWithLogitsLoss.
La tete de domaine est reinitialisee a chaque nouvelle adaptation.

Convention source=0, cible=1. L_total = L_RUL_source + L_domain avec
L_domain = 0.5*BCE_source + 0.5*BCE_cible, meme pour des batchs de tailles
differentes. La GRL est l'identite en forward et multiplie le gradient par
-lambda en backward. Pas de perte soustraite ni de lambda applique deux fois.
L'encodeur recoit gradient RUL - lambda*gradient domaine ; la tete de domaine
recoit le gradient normal de classification. Schedule logistique par defaut :
lambda_max*(2/(1+exp(-gamma*p))-1), p=global_step/max(1,total_updates-1).
Le premier update utilise lambda=0, le dernier approche lambda_max. Un schedule
constant est aussi disponible. Le lambda moyen par epoch est trace.

Sans `steps_per_epoch`, DANN utilise max(nombre_batches_source,
nombre_batches_cible). Le loader epuise est explicitement reitere. Toutes les
epochs ont ce budget, sans `zip` tronque au plus court. Les stats d'epoch sont des
moyennes par update (chaque perte domaine equilibre deja les deux domaines).

### Reprise et artefacts

```bash
cmapss train --config configs/source_fd001.yaml --output runs/source_s42 --resume runs/source_s42/checkpoints/last.pt
cmapss adapt --config configs/dann_fd001_to_fd002.yaml --source-checkpoint runs/source_s42/checkpoints/best.pt --output runs/dann_s42 --resume runs/dann_s42/checkpoints/last.pt
```

Reprise a une frontiere d'epoch terminee, depuis last.pt dans son repertoire
original, meme configuration resolue, memes fichiers et meme environnement.
Elle restaure poids, optimiseur, compteurs, early stopping, historique et RNG
Python/NumPy/Torch/CUDA/DataLoader. Scheduler/AMP sont explicitement absents
(et etats null). Pas de promesse de reprise au milieu d'un batch ni de
reproductibilite bit-a-bit entre plateformes/CUDA/versions differentes. En cas
d'interruption entre l'ecriture du CSV et celle du checkpoint, un decalage est
refuse plutot que masque. Un budget termine ne peut pas etre allonge par reprise
silencieuse ; lancer une nouvelle experience explicite.

Chaque run train/adapt contient configuration resolue, manifeste des moteurs,
preprocessing, SHA256 des entrees, environnement/versions/Git, history.csv,
checkpoints et PNG. `run_summary.json` donne le budget et la duree cumulee des
segments de calcul (hors preparation/figures). Les evaluations separees ajoutent
metrics.json, predictions.csv, mapping explicite RUL terminales et figures.
Les exports trajectoires sont distincts de predictions.csv et concernent les
plus petits IDs (`trajectory_count`, 5 par defaut), jamais les meilleurs cas.
Les CSV prediction conservent RUL brute, cible transformee, prediction et erreurs.
Les checkpoints PyTorch contiennent des objets Python : ne charger que des
checkpoints de confiance.

## Tests et validation de cette livraison

`tests/test_data.py` couvre parsing/erreurs, labels, split, preprocessing, fenetres,
padding, metriques, inspection CLI et figures. `tests/test_torch.py` couvre
architecture, GRL/signe/amplitude, gradients encodeur et domaine, batchs inegaux,
entrainement minuscule, checkpoint, reprise, smoke CLI FD001, transfert sans train
cible et invariance DANN aux modifications des labels cibles d'evaluation.

Voir `VALIDATION.md` pour le bilan reel, distinct des tests seulement ecrits.
La CI CPU execute pytest puis le smoke test et archive ses artefacts. Ajouter une
CI ne signifie pas qu'elle a deja ete executee avec succes.

## Limites scientifiques

Une accuracy domaine proche de 50% n'est pas une preuve de bon transfert : elle
peut indiquer un discriminateur faible ou des representations non informatives.
Aligner des distributions de sante differentes peut degrader la regression ; le
DANN n'est pas garanti superieur au transfert direct. Le DANN complet entraine
encore l'encodeur et reutilise les donnees source : il n'est pas automatiquement
frugal en calcul ou en memoire. Ce pipeline n'est pas annonce comme une
reproduction exacte d'un article et aucun score attendu n'est promis.

References conceptuelles : Ganin et al., *Domain-Adversarial Training of Neural
Networks* ; da Costa et al., *Remaining Useful Lifetime Prediction via Deep
Domain Adaptation*. Elles motivent l'alignement adversarial, sans equivalence
revendiquee de protocole ou de resultats.
