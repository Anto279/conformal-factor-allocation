# Allocation entre facteurs actions

Projet de recherche construit étape par étape. Chaque étape répond à une question.

## Étape 1 : pourquoi s'intéresser aux modèles à facteurs ?

- **Les facteurs expliquent l'essentiel du risque d'un portefeuille diversifié.** Sur un titre isolé, le risque spécifique domine ; dès qu'on diversifie, il s'efface et il ne reste que l'exposition à quelques facteurs communs (marché, taille, value, momentum, qualité, faible volatilité). Comme le résume Roncalli (2017) : *« Common risk factors are the only bets that are compatible with diversification. »*
- **Une bonne partie de l'« alpha » est en fait du facteur.** La performance d'un gérant ou d'une stratégie se décompose en expositions factorielles plus un résidu, souvent faible. Raisonner en facteurs permet de savoir ce que l'on porte réellement.
- **Les facteurs sont des primes de risque, pas des prévisions.** Ils rémunèrent un risque supporté dans les mauvais états du monde ou exploitent une anomalie persistante. On peut donc les détenir sans savoir prédire leur rendement le mois suivant.
- **Prédire les rendements ne marche pas (ou presque).** Mon projet précédent, [strat-invest-direct](https://github.com/Anto279/strat-invest-direct), testait des prévisions de rendement (ARIMA, XGBoost) sur 8 ETF : elles ne font pas mieux que la moyenne historique (R² hors échantillon proche de zéro ou négatif), et la validation croisée leur attribue un poids nul. Le **risque**, lui, se prévoit : le GARCH bat significativement l'EWMA.
- **D'où l'ERC (Equal Risk Contribution).** Si l'on ne sait prévoir que le risque, il faut une allocation qui n'utilise que le risque. L'ERC donne à chaque facteur la même contribution au risque total, sans aucune hypothèse de rendement espéré. Roncalli (2017) la décrit comme *« a highly-regularized MVO portfolio »*, robuste et diversifiée.
- **C'est l'approche d'Amundi.** La gamme Dynamic Multi Factors combine des facteurs (value, momentum, qualité, faible volatilité, taille) avec une allocation stratégique en budget de risque, ajustée ensuite de façon tactique. La stratégie Global Dynamic Multi-Factor Equity a reçu le prix *Equity Multi-Factor Solution Provider of the Year* (PP Investment Awards 2024).

**Référence** : Roncalli T. (2017), *Alternative Risk Premia: What Do We Know?*, Amundi Working Paper WP-061-2017. [Lien](https://research-center.amundi.com/files/nuxeo/dl/e8c7f3a3-511c-4b68-a2fa-34d168f6a03d)
