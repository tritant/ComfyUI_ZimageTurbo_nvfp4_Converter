# ComfyUI Z-image-Turbo to NVFP4 converter

Un nœud simple et ultra-rapide pour convertir Z-image-turbo au format **NVFP4** directement depuis l'interface ComfyUI.

Ce format permet de diviser la taille des modèles par 3.5 tout en conservant une qualité quasi identique au BF16, tout en profitant des **Tensor Cores** des cartes NVIDIA récentes.
<img width="1750" height="808" alt="Capture d&#39;écran 2026-01-22 161324" src="https://github.com/user-attachments/assets/ae23a6db-e948-48aa-99a8-2005bad1a4d7" />

## 🛠️ Installation

1. **Prérequis** :
Assurez-vous d'avoir installé la bibliothèque `comfy-kitchen` dans l'environnement Python de votre ComfyUI :
```bash
pip install comfy-kitchen

```


2. **Installation du nœud** :
Allez dans votre dossier `custom_nodes` et clonez ce dépôt (ou via manager) :
```bash
cd custom_nodes
git clone https://github.com/tritant/ComfyUI_ZimageTurbo_nvfp4_Converter.git
```


3. **Redémarrez ComfyUI**.

## 📖 Utilisation

1. Cherchez le nœud **🍳 Kitchen NVFP4 Converter** dans la catégorie `Kitchen`.
2. Sélectionnez votre modèle source dans la liste `model_name`.
3. Choisissez un nom pour le fichier de sortie (ex: `mon_modele_nvfp4`).
4. Réglez le `device` sur **cuda** pour une vitesse maximale.
5. Appuyez sur **Queue Prompt**.
---
