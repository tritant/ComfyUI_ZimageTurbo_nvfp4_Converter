import os
import json
import torch
import folder_paths
import safetensors.torch
import comfy.utils  # Import indispensable pour la barre de progression

# Importation de la cuisine (installée via pip)
try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("⚠️ [Convert-to-NVFP4] comfy-kitchen introuvable. Installation requise : pip install comfy-kitchen")

class ConvertToNVFP4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": ("STRING", {"default": "Z-model-nvfp4"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "Kitchen"
    
    # Force ComfyUI à considérer le nœud comme une sortie (évite l'erreur 'no outputs')
    OUTPUT_NODE = True

    def convert(self, model_name, output_filename, device):
        input_path = folder_paths.get_full_path("diffusion_models", model_name)
        output_path = os.path.join(os.path.dirname(input_path), f"{output_filename}.safetensors")
        
        # Blacklist stricte originale
        BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]

        print(f"🚀 Chargement du modèle source : {model_name}")
        sd = safetensors.torch.load_file(input_path)
        
        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        
        # --- INITIALISATION DE LA BARRE DE PROGRESSION ---
        total_steps = len(sd)
        pbar = comfy.utils.ProgressBar(total_steps)
        # --------------------------------------------------

        print(f"⚙️ Conversion NVFP4 lancée sur le device : {device}")
        
        for i, (k, v) in enumerate(sd.items()):
            # Mise à jour de la barre bleue dans l'interface
            pbar.update_absolute(i + 1)

            # Respect de la Blacklist
            if any(name in k for name in BLACKLIST):
                new_sd[k] = v.to(dtype=torch.bfloat16)
                continue

            # Quantification des poids linéaires (2D)
            if v.ndim == 2 and ".weight" in k:
                base_k_file = k.replace(".weight", "")
                base_k_meta = base_k_file.replace("model.diffusion_model.", "")
                
                # Déplacement sur le device pour utiliser les kernels optimisés (Triton/CUDA)
                v_tensor = v.to(device=device, dtype=torch.bfloat16)
                
                # Exécution de la quantification
                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                
                # Retour sur CPU pour la sauvegarde Safetensors
                for suffix, tensor in tensors.items():
                    new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                
                quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                
                if device == "cuda":
                    del v_tensor
            else:
                new_sd[k] = v.to(dtype=torch.bfloat16)

        # Métadonnées pour le Loader Core
        metadata = {"_quantization_metadata": json.dumps(quant_map)}
        
        print(f"💾 Écriture du fichier final sur le disque...")
        safetensors.torch.save_file(new_sd, output_path, metadata=metadata)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"✅ Conversion terminée avec succès : {output_path}")
        return (f"Modèle créé sur {device} : {output_filename}.safetensors",)

NODE_CLASS_MAPPINGS = {"ConvertToNVFP4": ConvertToNVFP4}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertToNVFP4": "🍳 Z-image-Turbo NVFP4 Converter"}