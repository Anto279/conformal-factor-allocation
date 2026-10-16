# Allocation entre facteurs actions

Projet de recherche construit étape par étape. Chaque étape répond à une question.

## Étape 1 : pourquoi s'intéresser aux modèles à facteurs ?

- **Les facteurs expliquent l'essentiel du risque d'un portefeuille diversifié.** Sur un titre isolé, le risque spécifique domine ; dès qu'on diversifie, il s'efface et il ne reste que l'exposition à quelques facteurs communs (marché, taille, value, momentum, qualité, faible volatilité). Comme le résume Roncalli (2017) : *« Common risk factors are the only bets that are compatible with diversification. »*
- **Une bonne partie de l'« alpha » est en fait du facteur.** La performance d'un gérant ou d'une stratégie se décompose en expositions factorielles plus un résidu, souvent faible. Raisonner en facteurs permet de savoir ce que l'on porte réellement.
- **Les facteurs sont des primes de risque, pas des prévisions.** Ils rémunèrent un risque supporté dans les mauvais états du monde ou exploitent une anomalie persistante. On peut donc les détenir sans savoir prédire leur rendement le mois suivant.
- **Prédire les rendements ne marche pas (ou presque).** Mon projet précédent, [strat-invest-direct](https://github.com/Anto279/strat-invest-direct), testait des prévisions de rendement (ARIMA, XGBoost) sur 8 ETF : elles ne font pas mieux que la moyenne historique (R² hors échantillon proche de zéro ou négatif), et la validation croisée leur attribue un poids nul. Le **risque**, lui, se prévoit : le GARCH bat significativement l'EWMA.
- **D'où l'ERC (Equal Risk Contribution).** Si l'on ne sait prévoir que le risque, il faut une allocation qui n'utilise que le risque. L'ERC donne à chaque facteur la même contribution au risque total, sans aucune hypothèse de rendement espéré. Roncalli (2017) la décrit comme *« a highly-regularized MVO portfolio »*, robuste et diversifiée.
- **C'est l'approche d'Amundi.** La gamme Dynamic Multi Factors combine des facteurs (value, momentum, qualité, faible volatilité, taille) avec une allocation stratégique en budget de risque, ajustée ensuite de façon tactique. La stratégie Global Dynamic Multi-Factor Equity a reçu le prix *Equity Multi-Factor Solution Provider of the Year* (PP Investment Awards 2024).

## Étape 2 : le modèle Fama-French + momentum

### 2.1 La formule

Pour un actif ou un portefeuille $i$, au mois $t$ :

$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_i\,\mathrm{MKT}_t + s_i\,\mathrm{SMB}_t + h_i\,\mathrm{HML}_t + r_i\,\mathrm{RMW}_t + c_i\,\mathrm{CMA}_t + m_i\,\mathrm{MOM}_t + \varepsilon_{i,t}
$$

C'est le modèle à cinq facteurs de Fama et French (2015), complété par le facteur momentum de Carhart (1997).

En prenant l'espérance, et si le modèle est juste ($\alpha_i = 0$) :

$$
\mathbb{E}[R_i] - R_f = \beta_i\,\lambda_{\mathrm{MKT}} + s_i\,\lambda_{\mathrm{SMB}} + h_i\,\lambda_{\mathrm{HML}} + r_i\,\lambda_{\mathrm{RMW}} + c_i\,\lambda_{\mathrm{CMA}} + m_i\,\lambda_{\mathrm{MOM}}, \qquad \lambda_k = \mathbb{E}[F_k]
$$

Le rendement attendu d'un actif = somme de (son exposition à chaque facteur) × (la prime moyenne de ce facteur).

### 2.2 D'où vient cette formule

- **CAPM (Sharpe, 1964)** : un seul facteur, le marché. $\mathbb{E}[R_i] - R_f = \beta_i\,(\mathbb{E}[R_M] - R_f)$. Seul le risque non diversifiable est rémunéré.
- **Le CAPM échoue empiriquement** : les petites capitalisations (Banz, 1981) et les actions « value », bon marché par rapport à leurs fonds propres (Basu, 1977 ; Rosenberg et al., 1985), rapportent plus que ce que leur bêta prévoit.
- **Fama-French 3 facteurs (1993)** : on ajoute deux facteurs qui captent ces écarts, SMB (taille) et HML (value).
- **Carhart (1997)** : ajoute MOM, car les gagnants des 12 derniers mois continuent de surperformer (Jegadeesh et Titman, 1993), ce que les trois facteurs n'expliquent pas.
- **Fama-French 5 facteurs (2015)** : ajoute RMW (rentabilité) et CMA (investissement). Ces deux facteurs ne sont pas choisis au hasard : ils découlent du modèle d'actualisation des dividendes, réécrit avec les données comptables :

$$
\frac{M_t}{B_t} = \frac{\sum_{\tau \ge 1} \mathbb{E}\left[Y_{t+\tau} - dB_{t+\tau}\right] / (1+r)^{\tau}}{B_t}
$$

  $M$ = capitalisation, $B$ = fonds propres comptables, $Y$ = bénéfices, $dB$ = investissement (variation des fonds propres), $r$ = rendement attendu. À tout le reste égal, $r$ est plus élevé quand le prix est bas par rapport aux fonds propres (value), quand la rentabilité est forte (RMW) et quand l'investissement est faible (CMA).

### 2.3 Pourquoi ce modèle ici (lien avec l'étape 1)

- C'est le modèle de référence, en recherche comme en gestion : les facteurs sont publics, construits de façon transparente et disponibles depuis 1963 (bibliothèque de Kenneth French).
- Sur des portefeuilles diversifiés, il explique l'essentiel des variations de rendement : le $R^2$ des régressions dépasse souvent 90 % (Fama et French, 1993). C'est exactement l'idée de l'étape 1 : une fois diversifié, il ne reste que des facteurs.
- Chaque facteur est lui-même un **portefeuille investissable** (long-short). On peut donc les traiter comme des actifs et allouer entre eux, avec une méthode qui n'a besoin que de leur risque (ERC).
- Ils recouvrent les styles du processus Amundi : value (HML), momentum (MOM), qualité (proche de RMW et CMA), taille (SMB).

### 2.4 Les paramètres

| Symbole | Ce que c'est | Nature |
|---|---|---|
| $R_{i,t}$ | rendement de l'actif ou du portefeuille $i$ au mois $t$ | rendement (%) |
| $R_{f,t}$ | taux sans risque : bon du Trésor américain à 1 mois | rendement (%) |
| $R_{i,t} - R_{f,t}$ | rendement excédentaire : ce que l'actif rapporte au-delà du placement sans risque | rendement (%) |
| $\mathrm{MKT}, \mathrm{SMB}, \dots, \mathrm{MOM}$ | les facteurs : rendements de portefeuilles construits par Fama-French | rendement (%) |
| $\beta_i, s_i, h_i, r_i, c_i, m_i$ | expositions (« loadings ») : de combien bouge $R_i$ quand le facteur bouge de 1 % | sans unité, estimées par régression MCO sur l'historique |
| $\lambda_k = \mathbb{E}[F_k]$ | prime du facteur $k$ : son rendement moyen | rendement (%) |
| $\alpha_i$ | rendement non expliqué par les facteurs ; vaut 0 si le modèle est juste | rendement (%) |
| $\varepsilon_{i,t}$ | choc spécifique à l'actif $i$, de moyenne nulle | rendement (%) |

Point clé : **les facteurs sont des rendements, pas des caractéristiques brutes.** La capitalisation, le ratio valeur comptable / valeur de marché ou la rentabilité servent seulement à **trier** les actions pour construire les portefeuilles. Ce qui entre dans la régression, c'est le rendement mensuel de ces portefeuilles.

**La prime d'un facteur est-elle celle d'un portefeuille diversifié plutôt que d'un titre ? Oui, avec une nuance.**

- Oui, chaque facteur est le rendement d'un portefeuille très diversifié (des centaines à des milliers d'actions). Une prime factorielle est donc la prime d'un portefeuille, pas d'un titre.
- Le membre de gauche, $R_i$, peut en revanche être n'importe quoi : une action seule, un fonds, un portefeuille. Le modèle s'applique aussi à un titre seul, mais il l'explique mal ($R^2$ bien plus faible), car le terme spécifique $\varepsilon_i$ y pèse lourd. Sur un portefeuille diversifié, $\varepsilon$ se compense et le $R^2$ devient très élevé.
- Le risque spécifique $\varepsilon_i$ n'est pas rémunéré : il se diversifie gratuitement. Seule l'exposition aux facteurs donne droit à une prime.

### 2.5 Comment Fama et French calculent les facteurs

Les facteurs ne sont pas calculés sur notre portefeuille. Fama et French les calculent sur l'ensemble des actions américaines (NYSE, AMEX, NASDAQ, environ 4 000 titres) et publient chaque mois leur rendement sur le site de Kenneth French. On télécharge ces séries telles quelles.

**La recette, identique pour tous les facteurs**

1. On prend toutes les actions américaines.
2. On mesure pour chacune une caractéristique (sa taille, son prix par rapport à sa valeur comptable, sa rentabilité...).
3. On les classe selon cette caractéristique et on forme deux groupes : les 30 % du haut et les 30 % du bas. Les 40 % du milieu ne servent pas.
4. Le facteur du mois = rendement moyen d'un groupe − rendement moyen de l'autre.

Un facteur est donc le résultat d'un pari « j'achète un groupe, je vends l'autre ». Par exemple, si en mars les actions bon marché gagnent 2 % et les actions chères 0,5 %, le facteur value vaut +1,5 % en mars.

Trois précisions :

- **Les groupes sont refaits une fois par an, fin juin**, avec les comptes de l'année précédente (déjà publiés à cette date). Seul le momentum est refait chaque mois.
- **Chaque action pèse selon sa capitalisation** dans le rendement de son groupe.
- **Le tri est fait séparément chez les petites et chez les grandes entreprises**, puis on fait la moyenne des deux. Sinon, le facteur value serait en partie un pari sur la taille, car les actions bon marché sont souvent des petites entreprises. La frontière entre « petite » et « grande » est la médiane des capitalisations du NYSE.

**Les six facteurs**

| Facteur | Groupe acheté | Groupe vendu | Critère de tri |
|---|---|---|---|
| MKT (marché) | tout le marché | placement sans risque | aucun |
| SMB (taille) | petites entreprises | grandes entreprises | capitalisation boursière |
| HML (value) | actions bon marché | actions chères | fonds propres comptables / capitalisation |
| RMW (rentabilité) | entreprises très rentables | entreprises peu rentables | bénéfice d'exploitation / fonds propres |
| CMA (investissement) | entreprises qui investissent peu | entreprises qui investissent beaucoup | croissance du total de l'actif sur un an |
| MOM (momentum) | actions qui ont le plus monté | actions qui ont le plus baissé | rendement des 12 derniers mois, sans le dernier |

**MKT, le marché.** Rendement de toutes les actions américaines moins le taux d'un bon du Trésor à un mois. Pourquoi ça rapporte : les actions chutent pendant les récessions, au pire moment pour l'épargnant. Ce risque ne disparaît pas en diversifiant, donc il est rémunéré.

**SMB, la taille.** On sépare les entreprises en deux selon leur capitalisation : petites d'un côté, grandes de l'autre. SMB = rendement des petites − rendement des grandes. Pourquoi ça rapporte : les petites entreprises sont plus fragiles en période de crise et leurs actions s'échangent moins facilement. Les investisseurs demandent un surplus de rendement pour les détenir. Cet effet est faible depuis les années 1980.

**HML, la value.** Pour chaque entreprise, on divise ses fonds propres comptables (ce que vaut l'entreprise dans ses comptes) par sa capitalisation (ce qu'en dit la Bourse). Un ratio élevé signale une action bon marché, un ratio faible une action chère. HML = rendement des 30 % les moins chères − rendement des 30 % les plus chères. Pourquoi ça rapporte : les entreprises bon marché sont souvent en difficulté et souffrent plus quand l'économie se retourne, c'est un risque. Autre lecture : les investisseurs s'emballent pour les entreprises en forte croissance et les paient trop cher.

**RMW, la rentabilité.** Pour chaque entreprise, on calcule son bénéfice d'exploitation (chiffre d'affaires moins coûts de production, frais généraux et intérêts) divisé par ses fonds propres. RMW = rendement des entreprises les plus rentables − rendement des moins rentables. Pourquoi ça rapporte : à prix égal, une entreprise plus rentable produit plus de bénéfices pour ses actionnaires. Le marché semble sous-estimer combien cette rentabilité dure dans le temps.

**CMA, l'investissement.** Pour chaque entreprise, on regarde de combien son total de l'actif a augmenté sur la dernière année. Une forte hausse veut dire qu'elle investit beaucoup (usines, rachats d'entreprises...). CMA = rendement des entreprises qui investissent peu − rendement de celles qui investissent beaucoup. Pourquoi ça rapporte : une entreprise investit beaucoup quand l'argent lui coûte peu, c'est-à-dire quand ses actionnaires attendent un rendement faible. Les dirigeants ont aussi tendance à surinvestir, et le marché paie trop cher ces projets.

**MOM, le momentum.** Pour chaque action, on calcule son rendement sur les 12 derniers mois, en excluant le dernier mois (une action qui a beaucoup monté le mois dernier a tendance à rebaisser un peu juste après, effet de très court terme qu'on veut écarter). MOM = rendement des 30 % qui ont le plus monté − rendement des 30 % qui ont le plus baissé. Les groupes sont refaits chaque mois. Pourquoi ça rapporte : les investisseurs réagissent lentement aux nouvelles, donc les tendances se prolongent quelques mois. Ce n'est pas vraiment une prime de risque mais plutôt une anomalie de comportement (Roncalli, 2017). Elle peut s'effondrer brutalement quand le marché repart d'un coup, comme en 2009 (Daniel et Moskowitz, 2016).

## Étape 3 : la base de données

Tout le code et les résultats de cette étape sont dans le notebook [`notebooks/etape3_description_base.ipynb`](notebooks/etape3_description_base.ipynb) : chaque calcul est suivi de son résultat.

### 3.1 Ce que contient la base

- **Source** : bibliothèque de Kenneth French, deux fichiers téléchargés le 15 octobre 2026 (base CRSP d'août 2026), conservés dans [`data/raw/`](data/raw/).
- **Une ligne = un mois**, de **juillet 1963 à août 2026**, soit **758 mois**. La valeur d'une ligne est le rendement réalisé pendant le mois.
- **Une colonne = un facteur**, en **rendement mensuel en %** : `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom`. S'y ajoute `RF`, le taux sans risque mensuel, qui n'est pas un facteur.
- Le momentum est disponible depuis 1927, mais les cinq autres facteurs commencent en juillet 1963 : la base commune démarre donc à cette date.
- Aucune valeur manquante, aucun mois absent.
- Sur la période, le taux sans risque vaut en moyenne 4,35 % par an, avec un pic à 16,2 % en juin 1981 et un plancher à 0 % fin 2008.

### 3.2 Statistiques descriptives

| | Mkt-RF | SMB | HML | RMW | CMA | Mom |
|---|---:|---:|---:|---:|---:|---:|
| Rendement annualisé (%) | 7,2 | 2,2 | 3,5 | 3,0 | 2,9 | 7,2 |
| Volatilité annualisée (%) | 15,4 | 10,5 | 10,3 | 7,9 | 7,2 | 14,6 |
| Ratio de Sharpe | 0,47 | 0,21 | 0,34 | 0,38 | 0,41 | 0,49 |
| t-stat de la moyenne | 3,7 | **1,7** | 2,7 | 3,0 | 3,2 | 3,9 |
| Mois positifs (%) | 60 | 50 | 54 | 56 | 52 | 62 |
| Pire mois (%) | −23,2 (oct. 1987) | −15,5 (mars 2000) | −13,8 (mars 2020) | −18,9 (fév. 2000) | −7,1 (mai 2023) | −34,4 (avr. 2009) |
| Meilleur mois (%) | 16,1 (oct. 1974) | 18,5 (fév. 2000) | 12,9 (janv. 2022) | 13,0 (nov. 2000) | 9,0 (fév. 2001) | 18,1 (fév. 2000) |
| Asymétrie | −0,49 | 0,36 | 0,09 | −0,24 | 0,26 | **−1,29** |
| Kurtosis en excès | 1,7 | 3,1 | 2,2 | **10,4** | 1,3 | **9,5** |
| Autocorrélation à 1 mois | 0,04 | 0,05 | 0,16 | 0,14 | 0,13 | 0,04 |
| Perte maximale (%) | −55,8 | −56,5 | −57,8 | −41,8 | −27,6 | −57,8 |
| Période de la perte maximale | 11/1968 à 09/1974 | 07/1983 à 03/1999 | 12/2006 à 09/2020 | 08/1998 à 02/2000 | 12/2022 à 10/2025 | 11/2008 à 09/2009 |

Rendement annualisé $= 12\,\bar r$, volatilité annualisée $= \sqrt{12}\,\hat\sigma$, Sharpe $= \sqrt{12}\,\bar r/\hat\sigma$, t-stat $= \bar r / (\hat\sigma/\sqrt{T})$. Le Sharpe se calcule sans retrancher de taux sans risque, car `Mkt-RF` est déjà un rendement excédentaire et les autres facteurs sont des positions long-short autofinancées.

Ce qu'on en retient :

- **Cinq primes sur six sont significatives** (t-stat > 2). La taille fait exception : 2,2 % par an, t-stat de 1,7, et la moitié des mois seulement sont positifs. Sur 63 ans, on ne peut pas affirmer que la prime de taille existe.
- **Les facteurs long-short ont des Sharpe proches de celui du marché, avec deux fois moins de volatilité** (7 à 10 % par an contre 15 %). CMA a le meilleur Sharpe des facteurs de style (0,41) avec la volatilité la plus faible.
- **Aucun facteur ne suit une loi normale** : le test de Jarque-Bera rejette la normalité pour les six (p-valeur < 10⁻¹³). Le momentum est le cas extrême : asymétrie de −1,29 et kurtosis de 9,5, avec une perte de 34 % en un seul mois (avril 2009). C'est l'anomalie à krachs décrite par Roncalli (2017) à l'étape 1.
- **La kurtosis de RMW (10,4) vient presque entièrement de la bulle internet** : ses meilleur et pire mois tombent en 2000.
- **Les pertes maximales sont longues** : −58 % sur quatorze ans pour la value (2006 à 2020), −56 % sur seize ans pour la taille. Un investisseur concentré sur un seul facteur peut attendre plus d'une décennie avant de revenir à son sommet.
- **HML, RMW et CMA ont une légère autocorrélation** (0,13 à 0,16), contre quasiment zéro pour le marché et le momentum.

### 3.3 Évolution des facteurs

![Rendement cumulé des facteurs](docs/figures/facteurs_cumules.png)

- Sur 63 ans, 1 $ investi devient 45 $ avec le momentum et 44 $ avec le marché, mais seulement 3 à 7 $ avec les autres facteurs (échelle logarithmique). C'est cohérent avec leur volatilité plus faible.
- Le momentum progresse régulièrement jusqu'en 2008, puis perd plus de la moitié de sa valeur en 2009, au moment du rebond des marchés. Il n'a jamais retrouvé son sommet depuis.
- La value (HML) culmine fin 2006 et ne fait que baisser jusqu'en 2020 : c'est la « décennie perdue » de la value.

![Rendement glissant sur 5 ans](docs/figures/rendement_glissant_5ans.png)

Le rendement annualisé sur 5 ans glissants passe régulièrement en territoire négatif pour tous les facteurs. Le tableau par décennie (notebook) le confirme :

| Décennie | Mkt-RF | SMB | HML | RMW | CMA | Mom |
|---|---:|---:|---:|---:|---:|---:|
| 1960s | 4,5 | 9,6 | 2,3 | 1,4 | −1,0 | 10,5 |
| 1970s | 1,2 | 4,7 | 7,8 | −0,6 | 6,2 | 9,8 |
| 1980s | 8,5 | −0,4 | 5,9 | 4,9 | 5,6 | 9,0 |
| 1990s | 12,8 | −2,1 | 0,1 | 2,3 | 0,2 | 13,5 |
| 2000s | −1,7 | 7,1 | 7,9 | 8,3 | 6,6 | 1,0 |
| 2010s | 13,1 | −0,4 | −2,3 | 1,4 | 0,2 | 3,2 |
| 2020s | 13,1 | −1,6 | 2,1 | 2,7 | 0,4 | 2,7 |

*Rendement annualisé en %. Les années 1960 commencent en juillet 1963, les années 2020 s'arrêtent en août 2026.*

Les facteurs se relaient : dans les années 2000, le marché perd de l'argent pendant que value, rentabilité et investissement rapportent 7 à 8 % par an. Dans les années 2010, c'est l'inverse. Aucun facteur ne gagne à toutes les décennies, ce qui plaide pour les combiner plutôt que d'en choisir un.

![Volatilité glissante sur 3 ans](docs/figures/volatilite_glissante_3ans.png)

- **La volatilité n'est pas constante.** Sur 3 ans glissants, elle varie d'un facteur 3 (marché) à 6,6 (rentabilité) entre son minimum et son maximum.
- **Les pics sont communs** : 2000-2002 (bulle internet) pour tous les facteurs de style, 2009 pour le marché et le momentum, 2022 pour la value et l'investissement.
- Le risque évolue donc par régimes, ce qui rejoint le constat de l'étape 1 : la volatilité se prévoit, et une allocation qui s'en sert doit l'estimer en continu plutôt qu'une fois pour toutes.

### 3.4 Corrélations

![Matrice de corrélation](docs/figures/matrice_correlation.png)

Avec 758 mois, une corrélation est significative au seuil de 5 % dès que $|\rho| > 1{,}96/\sqrt{758} \approx 0{,}07$. La corrélation moyenne en valeur absolue entre deux facteurs n'est que de **0,18**.

**Pourquoi ces corrélations ? Lecture à partir de la construction.** Un facteur est la différence entre deux portefeuilles, $F = L - S$ (jambe longue moins jambe courte). Pour deux facteurs :

$$
\mathrm{Cov}(L_1 - S_1,\ L_2 - S_2) = \mathrm{Cov}(L_1, L_2) - \mathrm{Cov}(L_1, S_2) - \mathrm{Cov}(S_1, L_2) + \mathrm{Cov}(S_1, S_2)
$$

Si les deux jambes longues contiennent les mêmes actions, et les deux jambes courtes aussi, les termes positifs l'emportent et les facteurs sont corrélés. De même, si chaque jambe a une sensibilité $\beta$ au marché, $F = (\beta_L - \beta_S)\,\mathrm{MKT} + \dots$ : le facteur garde une exposition résiduelle au marché, de signe $\beta_L - \beta_S$.

- **HML et CMA : +0,68**, la seule corrélation forte. Les actions bon marché (jambe longue de HML) sont souvent des entreprises qui investissent peu (jambe longue de CMA), et les actions chères sont souvent celles qui croissent vite. Les portefeuilles se recouvrent, d'où la corrélation. C'est la raison pour laquelle Fama et French (2015) trouvent HML presque redondant une fois CMA ajouté.
- **Mkt-RF avec HML (−0,21), RMW (−0,19), CMA (−0,36)** : les entreprises bon marché, rentables ou prudentes ont un bêta plus faible que les entreprises chères, peu rentables ou agressives. On a donc $\beta_L < \beta_S$ et une corrélation négative avec le marché : ces facteurs ont un profil défensif.
- **Mkt-RF et SMB : +0,27** : les petites capitalisations ont un bêta plus élevé que les grandes ($\beta_L > \beta_S$).
- **SMB et RMW : −0,33** : les entreprises peu rentables sont plus souvent petites, donc la jambe courte de RMW recouvre la jambe longue de SMB.
- **HML et Mom : −0,19** : une action qui a beaucoup monté (gagnante, jambe longue de Mom) a vu sa capitalisation grimper, donc son ratio valeur comptable / capitalisation baisser : elle devient « chère » (jambe courte de HML). Le momentum achète mécaniquement ce que la value vend.
- **Paires non significatives** : SMB/HML (0,02), RMW/CMA (0,02), CMA/Mom (−0,02), RMW/Mom (0,06, p = 0,13). SMB/HML est proche de zéro par construction : le tri croisé taille × valeur neutralise la taille dans HML et la valeur dans SMB.

![Corrélations glissantes sur 5 ans](docs/figures/correlations_glissantes.png)

Les corrélations ne sont pas stables dans le temps. HML/CMA reste toujours positive (entre 0,06 et 0,87) car le lien est structurel. HML/Mom, en revanche, oscille entre −0,65 et +0,6 : la corrélation de long terme (−0,19) cache des régimes très différents, dont un pic à +0,6 fin 2005, juste avant la crise de 2008.

### 3.5 Analyse en composantes principales

L'ACP est faite sur les rendements centrés-réduits, c'est-à-dire sur la matrice de corrélation $C = V\Lambda V^\top$. La part de variance expliquée par la composante $k$ vaut $\lambda_k / 6$. Si les six facteurs étaient indépendants, chaque composante expliquerait 1/6 ≈ 17 %.

![ACP : variance expliquée et poids](docs/figures/acp_variance_poids.png)

| | CP1 | CP2 | CP3 | CP4 | CP5 | CP6 |
|---|---:|---:|---:|---:|---:|---:|
| Valeur propre | 1,95 | 1,45 | 1,01 | 0,71 | 0,60 | 0,27 |
| Variance expliquée (%) | 33 | 24 | 17 | 12 | 10 | 5 |
| Variance cumulée (%) | 33 | 57 | 74 | 85 | 95 | 100 |

- **Il n'y a pas de composante dominante** : la première n'explique que 33 % de la variance, et il faut cinq composantes pour atteindre 95 %. Les six facteurs portent donc des risques réellement différents.
- **Le critère de Kaiser** ($\lambda > 1$) retient trois composantes, la troisième étant tout juste au-dessus du seuil (1,01).
- **CP1 (33 %), l'axe « défensif contre marché »** : HML et CMA d'un côté, le marché (et un peu SMB) de l'autre. Elle oppose les entreprises bon marché et prudentes au risque de marché.
- **CP2 (24 %), l'axe « petites entreprises fragiles contre qualité »** : SMB et HML d'un côté, RMW et Mom de l'autre. Elle oppose les petites entreprises bon marché aux entreprises rentables qui montent.
- **CP3 (17 %), le momentum seul** : Mom pèse 0,76, RMW −0,47. Le momentum forme une dimension à part.
- **CP6 (5 %) est la différence entre HML et CMA** (poids −0,67 et +0,69). Cette composante a très peu de variance, c'est la traduction de la redondance entre value et investissement.

![Cercle des corrélations](docs/figures/acp_cercle_correlations.png)

Sur le cercle des corrélations, HML et CMA sont presque superposées et proches du bord : elles sont bien représentées dans ce plan et très corrélées entre elles. Le marché est à l'opposé, ce qui reflète les corrélations négatives vues plus haut. La flèche du momentum est courte (0,51) : il est mal représenté par ce plan et s'exprime surtout sur CP3.

### 3.6 Classification hiérarchique

On regroupe les facteurs avec la distance $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ (0 si $\rho = 1$, $\sqrt 2 \approx 1{,}41$ si $\rho = 0$, 2 si $\rho = -1$) et la méthode d'agrégation de Ward.

![Dendrogramme](docs/figures/dendrogramme.png)

- **HML et CMA fusionnent en premier, à 0,80**, la seule paire nettement sous le seuil d'indépendance ($\sqrt 2$).
- **Mkt-RF et SMB fusionnent à 1,21** : le bloc « directionnel », exposé au risque de marché.
- **RMW et Mom fusionnent à 1,37**, juste sous $\sqrt 2$ : leur corrélation est presque nulle (0,06). Ce regroupement vient surtout de ce qu'ils sont loin des autres, pas de ce qu'ils se ressemblent.
- On retrouve les mêmes familles que dans l'ACP : un bloc marché et taille, un bloc value et investissement, et deux facteurs (rentabilité, momentum) qui évoluent chacun de leur côté.

### 3.7 À retenir de l'étape 3

- La base couvre 758 mois (juillet 1963 à août 2026), sans donnée manquante.
- Cinq primes sur six sont significatives. La taille ne l'est pas.
- Les facteurs ont des rendements à queues épaisses, des pertes maximales longues, et aucun ne gagne à toutes les décennies.
- Leur volatilité et leurs corrélations changent fortement dans le temps.
- Hormis HML et CMA, les facteurs sont peu corrélés entre eux, et l'ACP ne trouve aucune composante dominante : combiner les facteurs diversifie réellement le risque.

## Références

- Banz R. (1981), The Relationship Between Return and Market Value of Common Stocks, *Journal of Financial Economics*.
- Basu S. (1977), Investment Performance of Common Stocks in Relation to Their Price-Earnings Ratios, *Journal of Finance*.
- Carhart M. (1997), On Persistence in Mutual Fund Performance, *Journal of Finance*.
- Daniel K., Moskowitz T. (2016), Momentum Crashes, *Journal of Financial Economics*.
- Fama E., French K. (1993), Common Risk Factors in the Returns on Stocks and Bonds, *Journal of Financial Economics*.
- Fama E., French K. (2015), A Five-Factor Asset Pricing Model, *Journal of Financial Economics*.
- Jegadeesh N., Titman S. (1993), Returns to Buying Winners and Selling Losers, *Journal of Finance*.
- Roncalli T. (2017), *Alternative Risk Premia: What Do We Know?*, Amundi Working Paper WP-061-2017. [Lien](https://research-center.amundi.com/files/nuxeo/dl/e8c7f3a3-511c-4b68-a2fa-34d168f6a03d)
- Rosenberg B., Reid K., Lanstein R. (1985), Persuasive Evidence of Market Inefficiency, *Journal of Portfolio Management*.
- Sharpe W. (1964), Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk, *Journal of Finance*.
- Kenneth French, *Data Library* : https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
