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
- **Fama-French 3 facteurs (1993)** : on ajoute deux portefeuilles qui captent ces écarts, SMB (taille) et HML (value).
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

### 2.5 Construction des facteurs (bibliothèque de Kenneth French, États-Unis)

**Règles communes**

- Univers : actions ordinaires cotées au NYSE, AMEX et NASDAQ (données CRSP et Compustat).
- Seuils de tri calculés **sur les seules actions du NYSE**, pour que les milliers de micro-capitalisations du NASDAQ ne déplacent pas les seuils.
- Les portefeuilles sont formés **fin juin de chaque année $t$**, conservés de juillet $t$ à juin $t+1$, et pondérés par la capitalisation. Les données comptables viennent de l'exercice clos en $t-1$, donc publiées avant la formation.
- Tri double : taille (Small / Big, coupure à la médiane NYSE) × caractéristique (coupures aux 30ᵉ et 70ᵉ centiles NYSE). On obtient 2 × 3 = 6 portefeuilles. Le facteur moyenne les petites et les grandes capitalisations, pour neutraliser l'effet taille.
- Résultat : des rendements mensuels, en %.

**MKT (marché)**

- Calcul : rendement pondéré par la capitalisation de toutes les actions de l'univers, moins le taux du T-Bill à 1 mois.
- Économie : c'est la prime de risque actions. Les actions perdent dans les récessions, au moment où l'épargne compte le plus ; ce risque ne se diversifie pas, il est donc rémunéré.

**SMB (Small Minus Big, taille)**

- Calcul : rendement moyen des petites capitalisations moins celui des grandes. Dans le modèle à 5 facteurs, c'est la moyenne de trois versions (taille croisée avec value, rentabilité et investissement) :

$$
\mathrm{SMB} = \tfrac{1}{3}\left(\mathrm{SMB}_{B/M} + \mathrm{SMB}_{OP} + \mathrm{SMB}_{INV}\right), \qquad \mathrm{SMB}_{B/M} = \tfrac{1}{3}(SH + SN + SL) - \tfrac{1}{3}(BH + BN + BL)
$$

- Économie : les petites entreprises sont plus fragiles (accès au crédit, sensibilité aux récessions) et moins liquides. Les investisseurs demandent une prime pour porter ce risque. Cette prime est faible depuis les années 1980.

**HML (High Minus Low, value)**

- Caractéristique de tri : $B/M$ = fonds propres comptables de l'exercice $t-1$ / capitalisation fin décembre $t-1$.

$$
\mathrm{HML} = \tfrac{1}{2}(SH + BH) - \tfrac{1}{2}(SL + BL)
$$

  $H$ = 30 % de $B/M$ les plus élevés (actions bon marché), $L$ = 30 % les plus bas (actions chères, « growth »).
- Économie : explication par le risque, les entreprises value sont souvent en difficulté, avec des actifs lourds difficiles à réduire quand l'économie se retourne ; elles souffrent le plus dans les mauvais états du monde. Explication comportementale, les investisseurs extrapolent trop la croissance passée, surpaient les « growth » et délaissent les value (Lakonishok, Shleifer et Vishny, 1994).

**RMW (Robust Minus Weak, rentabilité)**

- Caractéristique de tri : rentabilité opérationnelle de l'exercice $t-1$,

$$
OP = \frac{\text{chiffre d'affaires} - \text{coût des ventes} - \text{frais généraux (SG\&A)} - \text{charges d'intérêts}}{\text{fonds propres comptables}}
$$

$$
\mathrm{RMW} = \tfrac{1}{2}(SR + BR) - \tfrac{1}{2}(SW + BW)
$$

  $R$ = rentabilité forte (30 % du haut), $W$ = faible (30 % du bas).
- Économie : d'après la formule de 2.2, à prix égal, une entreprise plus rentable génère plus de flux pour l'actionnaire, donc offre un rendement attendu plus élevé. Côté comportemental, le marché sous-estime la persistance de la rentabilité (Novy-Marx, 2013). C'est le cœur du facteur « qualité ».

**CMA (Conservative Minus Aggressive, investissement)**

- Caractéristique de tri : croissance du total de l'actif entre les exercices $t-2$ et $t-1$,

$$
INV = \frac{TA_{t-1} - TA_{t-2}}{TA_{t-2}}
$$

$$
\mathrm{CMA} = \tfrac{1}{2}(SC + BC) - \tfrac{1}{2}(SA + BA)
$$

  $C$ = investissement faible (30 % du bas), $A$ = agressif (30 % du haut).
- Économie : une entreprise investit beaucoup quand son coût du capital est bas, c'est-à-dire quand le rendement attendu par ses actionnaires est faible (théorie du $q$ : Cochrane, 1991 ; Hou, Xue et Zhang, 2015). Côté comportemental, les dirigeants qui investissent beaucoup ont tendance à surinvestir (« empire building »), ce que le marché paie trop cher.

**MOM (momentum, ou UMD : Up Minus Down)**

- Caractéristique de tri : rendement cumulé de $t-12$ à $t-2$. Le dernier mois est exclu, car il présente un effet de retournement à court terme (liquidité, fourchette achat-vente).
- Différence avec les autres facteurs : les portefeuilles sont **reformés chaque mois**, pas chaque année.

$$
\mathrm{MOM} = \tfrac{1}{2}(S\,\mathrm{High} + B\,\mathrm{High}) - \tfrac{1}{2}(S\,\mathrm{Low} + B\,\mathrm{Low})
$$

  High = 30 % des meilleurs rendements passés, Low = 30 % des pires.
- Économie : l'explication est surtout comportementale. L'information se diffuse lentement et les investisseurs sous-réagissent aux nouvelles, puis suivent la tendance (Hong et Stein, 1999). Aucune explication par le risque ne fait consensus. Au sens de Roncalli (2017), c'est une **anomalie** plutôt qu'une prime de risque pure, avec des krachs brutaux lors des rebonds de marché (2009 : Daniel et Moskowitz, 2016).

## Références

- Banz R. (1981), The Relationship Between Return and Market Value of Common Stocks, *Journal of Financial Economics*.
- Basu S. (1977), Investment Performance of Common Stocks in Relation to Their Price-Earnings Ratios, *Journal of Finance*.
- Carhart M. (1997), On Persistence in Mutual Fund Performance, *Journal of Finance*.
- Cochrane J. (1991), Production-Based Asset Pricing and the Link Between Stock Returns and Economic Fluctuations, *Journal of Finance*.
- Daniel K., Moskowitz T. (2016), Momentum Crashes, *Journal of Financial Economics*.
- Fama E., French K. (1993), Common Risk Factors in the Returns on Stocks and Bonds, *Journal of Financial Economics*.
- Fama E., French K. (2015), A Five-Factor Asset Pricing Model, *Journal of Financial Economics*.
- Hong H., Stein J. (1999), A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets, *Journal of Finance*.
- Hou K., Xue C., Zhang L. (2015), Digesting Anomalies: An Investment Approach, *Review of Financial Studies*.
- Jegadeesh N., Titman S. (1993), Returns to Buying Winners and Selling Losers, *Journal of Finance*.
- Lakonishok J., Shleifer A., Vishny R. (1994), Contrarian Investment, Extrapolation, and Risk, *Journal of Finance*.
- Novy-Marx R. (2013), The Other Side of Value: The Gross Profitability Premium, *Journal of Financial Economics*.
- Roncalli T. (2017), *Alternative Risk Premia: What Do We Know?*, Amundi Working Paper WP-061-2017. [Lien](https://research-center.amundi.com/files/nuxeo/dl/e8c7f3a3-511c-4b68-a2fa-34d168f6a03d)
- Rosenberg B., Reid K., Lanstein R. (1985), Persuasive Evidence of Market Inefficiency, *Journal of Portfolio Management*.
- Sharpe W. (1964), Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk, *Journal of Finance*.
- Kenneth French, *Data Library* : https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
