# Allocation dynamique entre facteurs : quand s'écarter de la parité de risque ?

**Question.** Je sais prévoir le risque des facteurs actions, beaucoup moins leur
rendement. Peut-on partir d'une allocation en parité de risque entre facteurs
et ne s'en écarter que lorsque la prévision de rendement est *mesurablement
fiable*, en utilisant la Conformal Prediction comme jauge de confiance ?

Projet de recherche personnel, mené sur six semaines avec un protocole écrit
et figé avant le premier test. Données publiques uniquement (Kenneth French
Data Library).

> Avancement : semaine 1 en cours (protocole, données, test go/no-go).

---

## 1. D'où vient ce projet

Mon projet précédent, [strat-invest-direct](https://github.com/Anto279/strat-invest-direct),
construisait une allocation hebdomadaire sur 8 ETF multi-actifs (actions,
obligations, or, matières premières, immobilier) à partir de prévisions
ARIMA + XGBoost, de volatilités GARCH et d'une optimisation moyenne-variance,
testée en walk-forward de 2011 à 2026 avec exécution le jour suivant et frais.

Il a conduit à deux conclusions nettes :

| Ce qui ne marchait pas | Ce qui marchait |
|---|---|
| **Prévoir le rendement.** R² hors échantillon proche de zéro ou négatif sur les 8 actifs ; la validation croisée purgée a appris d'elle-même à donner un **poids nul** aux prévisions. | **Prévoir le risque.** Le GARCH bat une moyenne exponentielle (test de Diebold-Mariano, p = 0,015) et le pilotage par budget de volatilité réduit nettement les pertes extrêmes : perte maximale de −12 % contre −21 % pour un 60/40. |

Le portefeuille final (Sharpe 0,66 contre 0,83 pour le 60/40) protégeait mieux
mais ne rapportait pas plus. La leçon : **sans prévision de rendement fiable, la
valeur vient de la gestion du risque**.

Ce projet en tire les conséquences :

1. **Changer d'univers.** Au lieu d'actifs dont l'espérance de gain dépend
   entièrement de la prévision, j'alloue entre **primes factorielles**
   (taille, value, rentabilité, investissement, momentum) documentées depuis
   des décennies et rémunérées en moyenne sans rien prévoir.
2. **Bâtir sur ce qui marche.** Le socle est la **parité de risque** entre
   facteurs, qui n'utilise que les volatilités prévues par GARCH.
3. **Ne s'écarter du socle que quand c'est justifié.** Plutôt que de suivre une
   prévision de rendement à la lettre (ce qui avait échoué), je mesure sa
   fiabilité date par date avec la **Conformal Prediction**, et je ne dévie que
   lorsque la fourchette est étroite et loin de zéro.

## 2. Le lien avec la gestion multi-facteurs d'Amundi

En 2024, Amundi a reçu le prix *Equity Multi-Factor Solution Provider of the
Year* (Professional Pensions Investment Awards) pour sa stratégie **Global
Dynamic Multi-Factor Equity**. D'après les documents publics de l'équipe
(Amundi Viewpoint 2017, communiqué de lancement 2018), le processus repose sur :

- une allocation **stratégique en contributions au risque égales (ERC)** entre
  facteurs (value, momentum, qualité, faible volatilité, taille) ;
- des **ajustements dynamiques** de cette allocation selon le régime de marché
  et la valorisation des facteurs ;
- une construction de portefeuille sous contraintes (secteurs, pays, coûts).

Mon projet se place exactement à la deuxième étape. Il ne reproduit pas le
modèle d'Amundi, qui n'est pas public : il part du **même socle public**, la
parité de risque entre facteurs, et pose la question que tout ajustement
dynamique doit résoudre :

> *À quel moment un signal est-il assez fiable pour justifier de s'écarter de
> l'allocation de référence, compte tenu des frais et d'un budget de risque
> limité ?*

La méthode proposée est générique : elle s'applique à n'importe quel signal
(régime, valorisation, momentum de facteurs, signal d'apprentissage
automatique), puisqu'elle ne porte pas sur la prévision elle-même mais sur
l'usage de son incertitude.

## 3. L'idée en une image

Pour le mois prochain, le modèle prévoit +0,8 % pour le facteur momentum.

| | Fourchette à 80 % | Ce qu'elle dit | Décision |
|---|---|---|---|
| Cas A | [−2,5 % ; +4,1 %] | contient zéro : le modèle ne sait pas | rester sur la parité de risque |
| Cas B | [+0,3 % ; +1,6 %] | étroite et entièrement positive | surpondérer le momentum, dans un budget fixé |

Même prévision centrale, décisions opposées : c'est la valeur ajoutée de
l'incertitude. Le test central du projet compare donc **la même prévision
utilisée avec et sans fourchette**.

## 4. Le cadre mathématique

*Section complétée au fil des étapes.*

## 5. Le protocole

Le protocole complet est dans [docs/protocole.md](docs/protocole.md). Il est
écrit avant tout backtest et figé par le tag `protocole-v1`.

| Période | Rôle |
|---|---|
| 07/1963 - 12/1989 | exploration et calibration : tous les choix se font ici |
| 01/1990 - fin des données | test sous scellé, un seul passage après le tag |
| Developed, Europe, Japon | réplication, sans aucun réglage |

## 6. Semaine 1 : protocole et données

| # | Étape | Statut |
|---|---|---|
| 1 | Repo et brouillon du protocole | fait |
| 2 | Copie figée et datée des données brutes | en cours |
| 3 | Construction de la base propre | à faire |
| 4 | Tests de cohérence | à faire |
| 5 | Variables décalées et test anti-fuite | à faire |
| 6 | Test go/no-go sur 1963-1989 | à faire |
| 7 | Protocole figé (`protocole-v1`) | à faire |

## Reproduire

```bash
pip install -r requirements.txt
python -m src.data.freeze_raw <dossier_des_zips> --date AAAA-MM-JJ   # étape 2
python -m src.data.build_factors                                     # étape 3
pytest                                                               # étapes 4 et 5
python -m scripts.describe_data
python -m scripts.go_no_go                                           # étape 6
python -m scripts.power                                              # étape 7
```
