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
