import json
import hashlib
import time
import os
from typing import Dict, Any, List
import logging
from huggingface_hub import list_repo_commits

class GeneRecordManager:
    def __init__(self, json_file_path: str = 'gene_records.json', expression_registry_path: str = 'expression_registry.json'):
        self.json_file_path = json_file_path
        self.expression_registry_path = expression_registry_path
        self.records: Dict[str, Any] = {}
        self.expression_registry: Dict[str, List[str]] = {}  # Maps expression hash to list of miners who used it
        self._load_records()
        self._load_expression_registry()

    def _load_records(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                self.records = json.load(f)

    def _save_records(self):
        with open(self.json_file_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def _load_expression_registry(self):
        if os.path.exists(self.expression_registry_path):
            with open(self.expression_registry_path, 'r') as f:
                self.expression_registry = json.load(f)

    def _save_expression_registry(self):
        with open(self.expression_registry_path, 'w') as f:
            json.dump(self.expression_registry, f, indent=2)

    def _compute_expression_hash(self, expr) -> str:
        """Compute a unique hash for any expression"""
        return hashlib.sha256(str(expr).encode()).hexdigest()

    def add_record(self, miner_hotkey: str, gene_hash: str, timestamp: float, performance: float, expr=None, repo_name: str = None):
        self.records[miner_hotkey] = {
            'gene_hash': gene_hash,
            'timestamp': timestamp,
            'performance': performance
        }

        if expr is not None:
            expr_hash = self._compute_expression_hash(expr)
            if expr_hash not in self.expression_registry:
                created_at = list_repo_commits(repo_id=repo_name)[0].created_at.timestamp
                self.expression_registry[expr_hash] = {"earliest_timestamp":created_at, "earliest_hotkey":miner_hotkey, "score":performance} #Check when scoring that the earliest is the one in question
            
            

            # if miner_hotkey not in self.expression_registry[expr_hash]:
            #     self.expression_registry[expr_hash].append(miner_hotkey)
            self._save_expression_registry()

        self._save_records()

    def is_expression_duplicate(self, expr) -> bool:
        """Check if an expression has been used before by any miner"""
        expr_hash = self._compute_expression_hash(expr)
        return expr_hash in self.expression_registry


    def get_record(self, miner_hotkey: str) -> Dict[str, Any]:
        return self.records.get(miner_hotkey, None)

    def get_all_records(self) -> Dict[str, Dict[str, Any]]:
        return self.records

    def should_download(self, miner_hotkey: str, remote_gene_hash: str) -> bool:
        record = self.get_record(miner_hotkey)
        if record is None:
            return True
        return record['gene_hash'] != remote_gene_hash