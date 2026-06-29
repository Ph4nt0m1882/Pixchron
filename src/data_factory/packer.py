import os
import tarfile
from io import BytesIO

class WebDatasetPacker:
    def __init__(self, output_dir="datasets_ready", max_size_mb=1000):
        """
        Empaqueteur au standard WebDataset (.tar).
        max_size_mb: Taille maximale d'un fichier .tar avant d'en créer un nouveau.
        """
        self.output_dir = output_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.current_tar = None
        self.current_tar_path = ""
        self.current_size = 0
        self.shard_index = 0

    def _open_new_shard(self, bucket: str):
        if self.current_tar is not None:
            self.current_tar.close()
            
        # Création du dossier du bucket (safe_hf ou private_train) s'il n'existe pas
        bucket_dir = os.path.join(self.output_dir, bucket)
        os.makedirs(bucket_dir, exist_ok=True)
        
        self.current_tar_path = os.path.join(bucket_dir, f"pixchron_{bucket}_shard_{self.shard_index:06d}.tar")
        self.current_tar = tarfile.open(self.current_tar_path, "w")
        self.current_size = 0
        self.shard_index += 1
        print(f"Création d'un nouveau Shard WebDataset : {self.current_tar_path}")

    def add_sample(self, image_id: str, image_bytes: bytes, metadata_json_str: str, bucket: str):
        """
        Ajoute une paire Image+JSON à l'archive .tar en cours.
        """
        if self.current_tar is None or self.current_size >= self.max_size_bytes:
            self._open_new_shard(bucket)
            
        # Ajout de l'image PNG
        img_info = tarfile.TarInfo(name=f"{image_id}.png")
        img_info.size = len(image_bytes)
        self.current_tar.addfile(tarinfo=img_info, fileobj=BytesIO(image_bytes))
        
        # Ajout des métadonnées JSON
        json_bytes = metadata_json_str.encode('utf-8')
        json_info = tarfile.TarInfo(name=f"{image_id}.json")
        json_info.size = len(json_bytes)
        self.current_tar.addfile(tarinfo=json_info, fileobj=BytesIO(json_bytes))
        
        # Mise à jour de la taille estimée du shard
        self.current_size += len(image_bytes) + len(json_bytes)

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            print(f"Shard final fermé : {self.current_tar_path}")

if __name__ == "__main__":
    # Test local
    packer = WebDatasetPacker(output_dir="src/data_factory/output_shards", max_size_mb=10)
    packer.add_sample("test_001", b"fake_png_data", '{"desc":"test"}', "safe_hf")
    packer.close()
    print("Module Packer testé avec succès.")
