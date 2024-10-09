import logging 
import os 
import json 
import requests
import time
import tempfile
from abc import ABC, abstractmethod

from dml.gene_io import save_individual_to_json

class PushDestination(ABC):
    @abstractmethod
    def push(self, gene, commit_message):
        pass

class PushMixin:
    def push_to_remote(self, gene, commit_message):
        if not hasattr(self, 'push_destinations'):
            logging.warning("No push destinations defined. Skipping push to remote.")
            return

        for destination in self.push_destinations:
            destination.push(gene, commit_message)

class HuggingFacePushDestination(PushDestination):
    def __init__(self, repo_name):
        self.repo_name = repo_name

    def push(self, gene, commit_message):
        from huggingface_hub import HfApi, Repository

        if not self.repo_name:
            logging.info("No Hugging Face repository name provided. Skipping push to Hugging Face.")
            return

        # Create a temporary file to store the gene data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(save_individual_to_json(gene), temp_file)
            temp_file_path = temp_file.name

        try:
            if not os.path.exists(self.repo_name):
                Repository(self.repo_name, clone_from=f"https://huggingface.co/{self.repo_name}")
            
            repo = Repository(self.repo_name, f"https://huggingface.co/{self.repo_name}")
            repo.git_pull()
            
            api = HfApi()
            api.upload_file(
                path_or_fileobj=temp_file_path,
                path_in_repo=f"best_gene.json",
                repo_id=self.repo_name,
                commit_message=commit_message
            )
            logging.info(f"Successfully pushed gene to Hugging Face: {commit_message}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

class PoolPushDestination(PushDestination):
    def __init__(self, pool_url, wallet):
        self.pool_url = pool_url
        self.wallet = wallet

    def push(self, gene, commit_message):
        data = self._prepare_request_data("push_gene")
        data["gene"] = save_individual_to_json(gene)
        data["commit_message"] = commit_message
        
        response = requests.post(f"{self.pool_url}/push_gene", json=data)
        if response.status_code == 200:
            logging.info(f"Successfully pushed gene to pool: {commit_message}")
        else:
            logging.error(f"Failed to push gene to pool: {response.text}")

    def _prepare_request_data(self, message, timestamp = time.time()):
        return {
            "public_address": self.wallet.hotkey.ss58_address,
            "signature": self.wallet.hotkey.sign(message).hex(),
            "message": message,
            "timestamp":timestamp
        }