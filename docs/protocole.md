# Protocole de recherche

Statut : **brouillon (étape 1)**. Ce document est figé par le tag `protocole-v1`
à la fin de la semaine 1. Après ce tag, aucun choix listé ici ne peut être
modifié sans être signalé dans le [journal des essais](journal_des_essais.md)
et compté dans le Deflated Sharpe Ratio.

## 1. Question

Les rendements des facteurs actions sont très difficiles à prévoir, leur
risque beaucoup moins. Une allocation entre facteurs qui part de la parité de
risque et ne s'en écarte que lorsque la prévision de rendement est
**mesurablement fiable** (fourchette de Conformal Prediction étroite et loin de
zéro) fait-elle mieux, une fois les frais payés :

1. que la parité de risque seule ;
2. que la même prévision utilisée sans fourchette ?

La comparaison 2 est la plus importante : c'est la seule qui isole l'apport de
l'incertitude, indépendamment de la qualité du modèle de prévision.

## 2. Hypothèses testées

| | Hypothèse | Mesure | Seuil |
|---|---|---|---|
| H1 | Les fourchettes sont calibrées hors échantillon | couverture réalisée, globale et par régime de volatilité | à moins de 3 points du niveau visé |
| H2 | La fourchette améliore l'usage de la prévision | rendement mensuel de « avec CP » moins « sans CP », test apparié | t de Newey-West > 2 |
| H3 | La méthode bat la parité de risque après frais | rendement mensuel de « avec CP » moins ERC, test apparié | t de Newey-West > 2, Deflated Sharpe > 0,95 |

Critères intermédiaires, plus faciles à démontrer que H2 et H3 et annoncés à
l'avance parce que la puissance statistique est limitée (section 9) :
taux de bonnes décisions quand la règle se déclenche, rotation évitée par
rapport à la prévision brute, corrélation des déviations avec la volatilité.

## 3. Données et découpage

Source unique : Kenneth French Data Library, copie figée et datée dans
`data/raw/AAAA-MM-JJ/` avec empreintes SHA-256.

| Période | Rôle | Ce qui est permis |
|---|---|---|
| 07/1963 - 12/1989 | exploration et calibration | tout : choix de la règle, go/no-go, réglages |
| 01/1990 - fin des données | test, sous scellé | un seul passage, après le tag `protocole-v1`, en walk-forward à fenêtre croissante |
| Developed, Europe, Japon (depuis 1990) | réplication | même protocole, aucun réglage supplémentaire |

Les données de la période de test ne sont jamais chargées par le code
d'exploration : `src/split.py` coupe les séries au 31/12/1989 **avant** tout
calcul.

## 4. Univers

*À trancher en étape 7 :* marché inclus ou non dans l'allocation, version
« jambe longue » ou non.

## 5. Modèles

**Volatilité.** GARCH(1,1) sur rendements quotidiens, paramètres réestimés
chaque fin d'année sur fenêtre croissante, prévision de la variance des 21
jours suivants.

**Prévision du rendement normalisé** $z_{f,t} = r_{f,t+1} / \hat\sigma_{f,t}$ :

- marche 0 : moyenne historique de $z$ par facteur (référence nulle) ;
- marche 1 (principal) : ridge en panel commun aux facteurs, constante propre
  à chaque facteur, variables `mom` (tendance 12 mois rapportée à sa
  volatilité) et `vol` (volatilité récente relative à son passé) ;
- marche 2 (ablation, semaine 3) : XGBoost quantile, mêmes variables.

**Fourchette.** CP normalisée :
$\hat\mu_{f,t} \pm q_t\,\hat\sigma_{f,t}$, $q_t$ quantile conforme des scores
$|z - \hat z|$ passés ; variante ACI.

## 6. Règle de décision

$$w_{f,t} = \frac{w^{ERC}_{f,t}\,(1 + \kappa\, s_{f,t})}{\sum_g w^{ERC}_{g,t}\,(1 + \kappa\, s_{g,t})}$$

*À trancher en étape 7 :* forme du score $s$ (binaire ou continu), niveau de
couverture, horizon.

**Critère go/no-go fixé dès maintenant :** sur 1963-1989, la règle binaire
(« s'écarter seulement si la fourchette exclut zéro ») n'est retenue que si
elle se déclenche sur au moins 10 % des couples (mois, facteur). Sinon, on
passe au score continu. Ce test ne regarde aucune performance.

## 7. Benchmarks

1/N entre facteurs ; ERC ; ERC à volatilité ciblée ; momentum de facteurs ;
même prévision sans fourchette.

## 8. Frais

*À fixer en étape 7.*

## 9. Puissance statistique

*À calculer en étape 7.*

## 10. Ce qui est interdit

- Regarder une performance de stratégie sur 1990 et après avant le tag.
- Modifier une valeur de ce document après le tag sans l'inscrire au journal.
- Ne publier que les variantes qui marchent.
