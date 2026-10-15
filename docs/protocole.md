# Protocole de recherche, version 1

**Signé le 15/10/2026, tag `protocole-v1`.** Tous les choix ci-dessous ont été
faits sur la période d'exploration 1963-1989, sans avoir calculé une seule
performance de stratégie après 1989. Après ce tag, toute modification est
inscrite dans le [journal des essais](journal_des_essais.md) et comptée dans
le Deflated Sharpe Ratio.

## 1. Question

Les rendements des facteurs actions sont très difficiles à prévoir, leur
risque beaucoup moins. Une allocation entre facteurs qui part de la parité de
risque et ne s'en écarte qu'en proportion de la **fiabilité mesurée** de la
prévision de rendement (fourchette de Conformal Prediction) fait-elle mieux,
une fois les frais payés :

1. que la parité de risque seule ;
2. que la même prévision utilisée sans fourchette ?

La comparaison 2 est la plus importante : c'est la seule qui isole l'apport de
l'incertitude, indépendamment de la qualité du modèle de prévision.

## 2. Hypothèses et critères de succès

| | Hypothèse | Mesure | Seuil |
|---|---|---|---|
| H1 | Les fourchettes sont calibrées hors échantillon | couverture réalisée, globale et par tercile de volatilité prévue | à moins de 3 points de 80 % au global, de 6 points par tercile |
| H2 | La fourchette améliore l'usage de la prévision | rendement mensuel net « avec CP » moins « sans CP », test apparié | t de Newey-West (6 retards) > 2 |
| H3 | La méthode bat la parité de risque après frais | rendement mensuel net « avec CP » moins ERC, test apparié | t de Newey-West > 2 et Deflated Sharpe > 0,95 |

Critères intermédiaires, annoncés à l'avance parce que la puissance est
limitée (section 9) :

- **qualité du signal** : R² hors échantillon de la ridge contre la moyenne
  historique, corrélation de rang entre prévision et rendement réalisé ;
- **taux de bonnes décisions** : part des mois où le signe de la déviation
  $w - w^{ERC}$ coïncide avec le signe du rendement relatif du facteur ;
- **rotation évitée** : rotation de « avec CP » rapportée à « sans CP » ;
- **indépendance vis-à-vis de la volatilité** : corrélation, facteur par
  facteur, entre le score $s$ et la volatilité prévue.

## 3. Données et découpage

Source : Kenneth French Data Library, copie figée le 15/10/2026 (base CRSP
202608) dans `data/raw/2026-10-15/`, empreintes SHA-256 dans `SHA256SUMS`,
vérifiées par les tests.

| Période | Rôle | Ce qui est permis |
|---|---|---|
| 07/1963 - 12/1989 | exploration et calibration | tout : choix de la règle, go/no-go, réglages |
| 01/1990 - 08/2026 | test, sous scellé | un seul passage, en walk-forward à fenêtre croissante |
| Developed, Europe, Japon (11/1990 - 08/2026) | réplication | même code, mêmes paramètres, aucun réglage |

Les séries sont coupées au 31/12/1989 **avant** tout calcul d'exploration
(`src/split.py`). Pendant le test, chaque décision prise à la fin du mois $t$
n'utilise que les données jusqu'à $t$ et s'applique au rendement du mois $t+1$.

## 4. Univers

**Cinq facteurs de style** : SMB (taille), HML (value), RMW (rentabilité),
CMA (investissement), MOM (momentum). Le facteur marché (Mkt-RF) est
**exclu de l'allocation** : il porte une décision différente (le bêta), alors
qu'un gérant multi-facteurs alloue entre styles avec un bêta proche de 1.

**Région principale** : États-Unis. **Réplication** : Developed, Europe,
Japon, sans réglage. Les fichiers quotidiens régionaux seront ajoutés à la
copie figée avant la semaine 5, sous une nouvelle date.

**Version « jambe longue »** : en robustesse (semaine 5), pour HML et SMB, à
partir des 6 portefeuilles taille × valorisation déjà figés.

## 5. Modèles

### Volatilité

GARCH(1,1) sur rendements quotidiens, paramètres réestimés chaque fin d'année
sur fenêtre croissante. Volatilité prévue du mois suivant :
$\hat\sigma^2_{f,t} = \sum_{h=1}^{21} \mathbb{E}_t[\sigma^2_{f,t+h}]$.

### Prévision du rendement normalisé

Cible : $z_{f,t} = r_{f,t+1} / \hat\sigma_{f,t}$.

- **Marche 0** : moyenne historique de $z_f$.
- **Marche 1 (modèle principal)** : ridge en panel,
  $z_{f,t} = a_f + b_1\,\text{mom}_{f,t} + b_2\,\text{vol}_{f,t} + \varepsilon$,
  constantes $a_f$ non pénalisées, $\lambda = 1 \times n$ sur variables
  standardisées, réestimée chaque mois sur fenêtre croissante.
- **Marche 2 (ablation, semaine 3)** : XGBoost quantile, mêmes variables.

Variables, toutes connues en $t$ :
$\text{mom}_{f,t} = \sum_{k=0}^{11} r_{f,t-k} \big/ \sqrt{\sum_{k=0}^{11} RV^2_{f,t-k}}$ et
$\text{vol}_{f,t} = \log RV_{f,t} - \overline{\log RV_f}^{\,[t-60,\,t-1]}$.
L'écart de valorisation est abandonné : il n'existe que pour HML et casserait
le modèle commun.

### Fourchette

CP normalisée, niveau **80 %** :
$\hat\mu_{f,t} \pm q_t\,\hat\sigma_{f,t}$, où $q_t$ est le
$\lceil (n+1)(1-\alpha) \rceil$-ième plus petit score $|z - \hat z|$ des
**60 derniers mois** (5 facteurs, soit 300 scores). Variante de robustesse :
ACI avec $\gamma = 0{,}005$.

*Pourquoi 60 mois et pas tout l'historique* : en exploration, l'écart-type de
$z$ baisse de 1,9 (années 1960) à 1,3 (fin des années 1980). Calibrée sur tout
l'historique, la CP sur-couvre (85 % au lieu de 80 %) ; sur 60 mois, elle
couvre 82 %.

## 6. Règle de décision

Score de confiance continu :

$$s_{f,t} = \operatorname{clip}\!\left(\frac{\hat\mu_{f,t}}{q_t\,\hat\sigma_{f,t}},\,-1,\,1\right)
= \operatorname{clip}\!\left(\frac{\text{centre de la fourchette}}{\text{demi-largeur}},\,-1,\,1\right)$$

Poids :

$$w_{f,t} = \frac{w^{ERC}_{f,t}\,(1 + \kappa\, s_{f,t})}{\sum_g w^{ERC}_{g,t}\,(1 + \kappa\, s_{g,t})}, \qquad \kappa = 1.$$

Avec $\kappa = 1$, le poids d'un facteur tombe à zéro exactement quand la
borne haute de sa fourchette touche zéro, et double quand la borne basse
touche zéro. La règle continue rejoint donc la règle binaire aux extrêmes.

**Résultat du go/no-go (étape 6)** : sur 1975-1989, la règle binaire
(« s'écarter seulement si la fourchette exclut zéro ») se déclenche sur
**0 %** des 840 couples (mois, facteur), à 80 % comme à 50 % de couverture.
Le seuil fixé à l'avance était de 10 % : elle est **rejetée** au profit du
score continu.

**Horizon** : 1 mois. Des cibles à 6 ou 12 mois amélioreraient le rapport
signal sur bruit mais se chevaucheraient d'un mois à l'autre, ce qui casse
l'hypothèse d'échangeabilité de la CP. Non retenu.

## 7. Modèle de risque et benchmarks

**Covariance pour l'ERC** :
$\Sigma_t = D_t R_t D_t$, avec $D_t = \operatorname{diag}(\hat c_{f,t}\,\hat\sigma_{f,t})$
et $R_t$ la corrélation des rendements mensuels sur 60 mois.
$\hat c_{f,t}$ est l'écart-type de $z_f$ sur les 60 derniers mois : il corrige
le fait que la volatilité quotidienne cumulée sous-estime la volatilité
mensuelle (écart-type de $z$ de 1,6 en exploration, différent d'un facteur à
l'autre). ERC résolu numériquement (Maillard, Roncalli et Teiletche, 2010).

| Benchmark | Définition |
|---|---|
| 1/N | 20 % sur chaque facteur de style |
| ERC | contributions au risque égales, $\Sigma_t$ ci-dessus |
| ERC à volatilité ciblée | ERC mis à l'échelle pour viser 10 % de volatilité annuelle prévue, levier plafonné à 2 |
| Momentum de facteurs | ERC restreint aux facteurs dont le rendement sur 12 mois est positif (ERC complet si aucun) |
| Sans CP (ablation clé) | même règle avec $s = \operatorname{clip}(\hat\mu_{f,t} / m_{f,t}, -1, 1)$, $m_{f,t} = q_t \times$ moyenne de $\hat\sigma_f$ sur 60 mois : même échelle moyenne, sans normalisation date par date |
| Marche 0 + règle | même règle avec la moyenne historique comme prévision |

## 8. Frais

Coût de transaction de **10 points de base** par unité de variation de
poids, appliqué au rééquilibrage mensuel. Sensibilité annoncée : 0, 25 et 50
points de base. Limite assumée : les coûts internes des facteurs long-short
(rotation propre au momentum notamment) ne sont pas modélisés ; le résultat
est une preuve de concept de la règle d'allocation, pas la performance d'un
fonds.

## 9. Puissance statistique

Période de test : 440 mois, soit 36,7 ans. Ratio d'information annualisé
minimal pour un test bilatéral à 5 % : **0,32** avec 50 % de puissance,
**0,46** avec 80 %. Les surcouches tactiques publiées sont plutôt entre 0,1 et
0,3 : un résultat « non significatif » sur H2 ou H3 est le plus probable et
sera publié tel quel. D'où les critères intermédiaires de la section 2 et la
réplication régionale.

## 10. Tests et présentation

- Différences mensuelles appariées, t de Newey-West à 6 retards.
- Bootstrap par blocs de 12 mois (10 000 tirages) sur la différence de Sharpe.
- Deflated Sharpe Ratio avec le nombre total d'essais du journal.
- Sous-périodes : 1990-1999, 2000-2009, 2010-2019, 2020-2026 ; résultats
  d'après 2000 présentés à part (les facteurs ont été publiés avant).

## 11. Version minimale (fin de semaine 3)

États-Unis uniquement : 1/N, ERC, ERC à volatilité ciblée, règle avec CP,
règle sans CP. Le reste (XGBoost, ACI, régions, jambe longue) est du bonus.

## 12. Ce qui est interdit

- Regarder une performance de stratégie sur 1990 et après avant ce tag.
- Modifier une valeur de ce document sans l'inscrire au journal.
- Ne publier que les variantes qui marchent.
