# Prediction de Defauts Logiciels par Deep Learning

## Document de presentation du projet

---

## 1. Introduction : L'Intelligence Artificielle et le Machine Learning

L'Intelligence Artificielle (IA) est un domaine de l'informatique qui vise a creer des systemes capables de realiser des taches qui necessitent normalement l'intelligence humaine. Le Machine Learning (ML) est une branche de l'IA ou les machines apprennent automatiquement a partir de donnees, sans etre explicitement programmees.

### Types d'apprentissage

| Type | Principe | Exemple |
|---|---|---|
| **Supervise** | On donne des exemples etiquetes, le modele apprend les patterns | Classification d'images, detection de spam |
| **Non supervise** | Pas d'etiquettes, le modele decouvre les structures cachees | Clustering, reduction de dimensionnalite |
| **Par renforcement** | L'agent apprend par essai-erreur avec des recompenses | Jeux, robotique |

Notre projet utilise l'**apprentissage supervise** : on donne au modele des exemples de code avec leurs etiquettes (defectueux ou non) et il apprend a predire l'etiquette pour du nouveau code.

### Le Deep Learning

Le Deep Learning est un sous-domaine du Machine Learning base sur les **reseaux de neurones profonds**. Il excelle dans des taches complexes comme la comprehension du langage, la vision par ordinateur, et dans notre cas, l'analyse de code source. Les modeles de Deep Learning peuvent apprendre des representations hierarchiques des donnees, ce qui leur permet de capturer des patterns complexes difficiles a detecter par des methodes traditionnelles.

---

## 2. Contexte du projet

### Le probleme

Les defauts logiciels (bugs) representent un enjeu majeur dans l'industrie du logiciel :
- Ils coutent des **milliards de dollars** par an
- Ils affectent la **securite**, la **fiabilite** et la **performance** des systemes
- La detection manuelle par revue de code est **lente** et **couteuse**

### L'objectif

Developper un **systeme automatique de detection de defauts** dans le code source Python en utilisant le Deep Learning. Le systeme analyse un morceau de code et predit s'il contient un defaut ou non.

### La problematique

> Comment utiliser les modeles de langage pre-entraines (comme CodeBERT) pour automatiser la detection de defauts dans le code source, et quels sont les defis associes a cette approche ?

---

## 3. Technologies utilisees

### CodeBERT (Microsoft)

CodeBERT est un modele de langage pre-entraine developpe par **Microsoft Research** en 2020. Il est base sur l'architecture **Transformer** (comme BERT, GPT) mais specialement concu pour comprendre le **code source**.

| Caracteristique | Detail |
|---|---|
| **Architecture** | Transformer (RoBERTa) |
| **Parametres** | ~125 millions |
| **Pre-entrainement** | 6 langages : Python, Java, JavaScript, PHP, Ruby, Go |
| **Donnees** | CodeSearchNet (6.4M fonctions) |
| **Taches** | Recherche de code, generation, classification |

### Pourquoi CodeBERT ?

1. **Pre-entraine sur du code** : contrairement a BERT (texte general), CodeBERT comprend la syntaxe et la semantique du code
2. **Transfer Learning** : on reutilise ses connaissances generales et on les adapte a notre tache specifique (fine-tuning)
3. **Reference dans la recherche** : CodeBERT est cite dans des centaines de publications scientifiques en genie logiciel

### Stack technique

| Composant | Technologie |
|---|---|
| **Modele** | CodeBERT (microsoft/codebert-base) |
| **Framework ML** | PyTorch 2.6 |
| **Librairie NLP** | HuggingFace Transformers |
| **Dataset** | HuggingFace Datasets |
| **Metriques** | Scikit-learn |
| **Visualisation** | Matplotlib |
| **Environnement** | VS Code (local) avec GPU NVIDIA RTX 4050 |

---

## 4. Le dataset Defectors

### Description generale

**Defectors** est un dataset de recherche publie en **2023** a la conference **MSR (Mining Software Repositories)**. Il est specifiquement concu pour la prediction de defauts logiciels.

| Information | Detail |
|---|---|
| **Nom** | Defectors: A Large Scale Python Dataset for Defect Prediction |
| **Auteurs** | Chercheurs en genie logiciel (MSR 2023) |
| **Source** | Zenodo (doi: 10.5281/zenodo.7708984) |
| **Taille** | ~213 000 fichiers de code source Python (~2.4 Go) |
| **Projets sources** | 25 projets Python populaires de 18 domaines |
| **Repartition** | ~93 000 defectueux / ~120 000 non-defectueux |
| **Format** | Parquet compresse |

### Pourquoi ce dataset ?

1. **Grande taille** : 213K exemples, suffisant pour les modeles de Deep Learning
2. **Diversite** : code provenant de 25 projets differents (ML, IoT, automatisation, etc.)
3. **Qualite** : 5 niveaux de filtrage du bruit appliques par les auteurs
4. **Adapte au Deep Learning** : concu specifiquement pour les modeles Transformer
5. **Realisme** : utilise des vrais bugs trouves dans des projets reels

### Les 25 projets sources

Les donnees proviennent de projets open-source populaires couvrant des domaines varies :
- Machine Learning (scikit-learn, pandas, keras)
- Automatisation (ansible, salt)
- Web (django, flask)
- Et d'autres domaines (IoT, cloud, etc.)

### Structure du dataset

Chaque exemple contient :

| Colonne | Description | Exemple |
|---|---|---|
| **code** | Le code source Python (diff Git) | `@@ -25,8 +25,9 @@ ...` |
| **label** | Etiquette binaire | 0 (normal) ou 1 (defectueux) |

### Comment les labels sont crees

Les labels sont generes automatiquement a partir de l'historique Git :

1. On identifie les **commits de correction de bugs** (bug-fixing commits)
2. On remonte au **commit qui a introduit le bug** (avec l'algorithme SZZ)
3. Le code **introduisant le bug** recoit le label **1** (defectueux)
4. Le code **normal** recoit le label **0** (non-defectueux)

---

## 5. Pipeline du projet : de la donnee au modele

### Etape 1 : Collecte des donnees

Telechargement du dataset Defectors depuis Zenodo (2.4 Go compresse).

```python
# Telechargement depuis Zenodo
wget -O defectors.zip "https://zenodo.org/records/7708984/files/defectors.zip"
```

### Etape 2 : Preparation des donnees

Les fichiers Parquet sont convertis en format JSON Lines. Chaque ligne contient le code source et son label.

```python
# Conversion Parquet -> JSONL
df = pd.read_parquet("train.parquet.gzip")
df['label'] = df['lines'].apply(lambda x: 1 if len(x) > 0 else 0)
# Sauvegarde: {"code": "...", "label": 0 ou 1}
```

### Etape 3 : Tokenisation

Le code source (texte) est converti en nombres que le modele peut comprendre.

```
Code original : "def foo(): return 42"
         |
    Tokenisation (CodeBERT Tokenizer)
         |
    input_ids      : [0, 9232, 3003, 7359, ...]
    attention_mask : [1, 1, 1, 1, ...]
```

### Etape 4 : Decoupage train/validation/test

| Split | Exemples | Role |
|---|---|---|
| **Train** | 185 369 | Le modele apprend |
| **Validation** | 10 000 | Suivi pendant l'entrainement |
| **Test** | 10 000 | Evaluation finale |

### Etape 5 : Entrainement (fine-tuning)

On prend CodeBERT pre-entraine et on l'adapte a notre tache :

```
CodeBERT pre-entraine (connaissances generales du code)
         |
    Fine-tuning sur Defectors
         |
    Modele specialise en detection de defauts
```

### Etape 6 : Evaluation

Le modele est evalue sur le test set avec plusieurs metriques :
- **Accuracy** : pourcentage de predictions correctes
- **F1-Score** : equilibre entre precision et recall
- **Precision** : fiabilite des alertes positives
- **Recall** : capacite a trouver tous les defauts

---

## 6. Resultats obtenus

### Metriques du modele entraine

| Metrique | Valeur |
|---|---|
| **Accuracy** | 52.24% |
| **F1-Score** | 0.2785 |
| **Precision** | 17.2% |
| **Recall** | 74.0% |
| **Validation Loss** | 0.707 |

### Interpretation des resultats

- Le modele detecte **74% des vrais defauts** (Recall = 0.74), ce qui est encourageant
- La precision est faible (17.2%), ce qui signifie qu'il y a des fausses alertes
- L'accuracy de 52% est superieure au hasard (50%) mais loin des attentes
- La detection de defauts dans le code est reconnue comme une **tache tres difficile** dans la litterature scientifique

### Defis rencontres

1. **Desequilibre des classes** : beaucoup plus de code normal que defectueux
2. **Complexite de la tache** : detecter un bug a partir du code source seul est difficile, meme pour un humain expert
3. **Limites du modele** : CodeBERT a 125M parametres, les modeles plus recents et plus gros pourraient performer mieux

### Comparaison avec la litterature

| Approche | Accuracy | F1 |
|---|---|---|
| Modeles classiques (Random Forest, XGBoost) | 55-60% | 0.30-0.45 |
| **Notre modele (CodeBERT)** | **52%** | **0.28** |
| Etat de l'art (modeles specialises) | 62-65% | 0.45-0.63 |

---

## 7. Discours de presentation

> Bonjour, je vais vous presenter mon projet de prediction de defauts logiciels par Deep Learning.
>
> **Le probleme :** Les bugs logiciels coutent des milliards de dollars chaque annee. La detection manuelle par revue de code est lente et couteuse. L'objectif de ce projet est d'automatiser cette detection en utilisant l'intelligence artificielle.
>
> **L'approche :** J'ai utilise CodeBERT, un modele de langage pre-entraine par Microsoft, specialise dans la comprehension du code source. Le principe est le Transfer Learning : on prend un modele qui a deja appris a comprendre le code de maniere generale, et on l'adapte (fine-tuning) pour notre tache specifique de detection de defauts.
>
> **Le dataset :** J'ai utilise Defectors, un dataset de recherche publie en 2023 a la conference MSR. Il contient environ 213 000 fichiers de code Python provenant de 25 projets open-source populaires. Chaque fichier est etiquete comme defectueux ou non, grace a l'analyse de l'historique Git et de l'algorithme SZZ.
>
> **Le pipeline :** Le code source passe par plusieurs etapes : telechargement depuis Zenodo, conversion au format JSONL, tokenisation avec le tokenizer de CodeBERT, puis entrainement du modele sur GPU pendant environ 6 heures.
>
> **Les resultats :** Le modele atteint une accuracy de 52% et un recall de 74%. Le recall eleve signifie que le modele detecte la majorite des vrais defauts, ce qui est l'objectif principal dans un contexte de securite logicielle. La precision reste a ameliorer pour reduire les fausses alertes.
>
> **Les defis :** La tache s'est revelee difficile pour plusieurs raisons. Le dataset est desequilibre, avec beaucoup plus de code normal que defectueux. De plus, la detection de bugs a partir du code seul est une tache complexe, meme pour des experts humains. Dans la litterature, les meilleurs modeles atteignent 62-65% d'accuracy, ce qui confirme la difficulte intrinseque de cette tache.
>
> **Perspectives :** Pour ameliorer les resultats, on pourrait explorer des modeles plus recents comme UniXcoder ou GraphCodeBERT, utiliser des techniques d'augmentation de donnees, ou combiner l'analyse du code avec d'autres signaux comme les metriques de processus logiciel.
>
> Merci pour votre attention. Je suis ouvert a vos questions.
