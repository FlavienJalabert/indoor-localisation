# Indoor Localization using WiFi and Inertial Sensors

> Projet de Master 2 ETAI – Polytech
> Indoor Localization / Machine Learning / Embedded Systems

---

## Description

Ce dépôt contient un projet de **localisation indoor** basé sur des signaux **WiFi** et des **capteurs inertiels**.  
L’objectif est d’étudier la faisabilité et les limites d’approches **data-driven** pour estimer une position 2D en environnement intérieur.

Le projet compare :
- des **modèles pointwise (statiques)**,
- des **modèles séquentiels (LSTM / GRU)**,
- ainsi qu’une **hybridation simple** entre les deux,

et analyse leur **robustesse inter-sessions et inter-devices** (ESP32 ↔ smartphone).

Ce travail s’inscrit dans un cadre académique (M2 ETAI) et vise avant tout une **analyse méthodologique rigoureuse**, plutôt que des performances état-de-l’art.

---

## Objectifs du projet

- Estimer la position 2D (X, Y) à partir de signaux WiFi et capteurs inertiels.
- Comparer approches statiques et séquentielles.
- Étudier la généralisation cross-device.
- Identifier les limites structurelles des approches purement data-driven en localisation indoor.

---

## Données

Les données correspondent à plusieurs trajectoires indoor enregistrées avec :
- mesures WiFi (RSSI),
- capteurs inertiels (accéléromètre, magnétomètre, Gyroscope),
- timestamps absolus,
- positions de référence (labels X, Y).

Chaque trajectoire est associée à :
- un appareil (`ESP32` ou `Samsung`),
- un contexte de mouvement (`motion`).

---

## Méthodologie

### Feature Engineering
- Sélection et encodage des points d’accès WiFi (top-k, présence, RSSI).
- Utilisation des signaux inertiels bruts avec dérivées simples et statistiques glissantes.
- Encodage des variables contextuelles (`motion`).
- Exclusion contrôlée de `device` lors des tests cross-device.

### Modèles
- **Pointwise** : Random Forest, XGBoost, kNN 
- **Séquentiels** : LSTM, GRU  
- **Hybride naïf** : combinaison linéaire XGB / LSTM

### Évaluation
- Tests cross-device.
- Splits par trajectoire (pas de fuite temporelle).
- Métriques : médiane, p90 / p95, CDF de l’erreur.
- Visualisation des trajectoires et des erreurs.

---

## Résultats principaux

- Les modèles pointwise offrent une meilleure stabilité spatiale d'après RMSE.
- Les modèles séquentiels améliorent la continuité temporelle mais dérivent spatialement (accumulation des erreurs).
- L’hybridation naïve montre un potentiel mais reste limitée (méthode trop simpliste).
- La généralisation cross-device est fortement impactée par le domain shift (offsets / bias).
- Sans carte ou contrainte spatiale explicite, un plafond de performance est atteint.

---

## Author

Master’s student – Embedded Systems & Artificial Intelligence
Flavien Jalabert
Polytech Nantes

---

## License

This project is provided for academic and educational purposes.
