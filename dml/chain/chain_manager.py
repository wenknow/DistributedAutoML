import bittensor as bt
from pydantic import BaseModel, Field
import functools
from typing import Optional, ClassVar, Type
import base64
import logging

import multiprocessing
import os
import lzma
import base64
import multiprocessing
from typing import Optional, Any
import logging

from bittensor.core.chain_data import decode_account_id

def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)

def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result

def decode_metadata(encoded_ss58: tuple, metadata: dict) -> tuple[str, str]:
    decoded_key = decode_account_id(encoded_ss58[0])
    block = metadata['block']
    commitment = metadata["info"]["fields"][0][0]
    bytes_tuple = commitment[next(iter(commitment.keys()))][0]
    return decoded_key, bytes(bytes_tuple).decode(), block



SHA256_BASE_64_LENGTH = 44  # Length of base64 encoded SHA256 hash
GIT_COMMIT_LENGTH = 40  # Length of git commit hash
MAX_METADATA_BYTES = 128  # Maximum metadata size allowed on chain

class SolutionId(BaseModel):
    """Uniquely identifies a solution/loss function"""
    class Config:
        frozen = True
        extra = "forbid"
    
    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES - GIT_COMMIT_LENGTH - SHA256_BASE_64_LENGTH - 2  # separators
    )
    
    repo_name: str = Field(
        description="Repository name where the solution can be found"
    )

    solution_hash: Optional[str] = Field(
        description="Hash of the solution/loss function",
        default=None
    )

    
    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.repo_name}:{self.solution_hash}"
    
    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["SolutionId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            repo_name=tokens[0],
            solution_hash=tokens[1] if tokens[1] != "None" else None
        )

class SolutionMetadata(BaseModel):
    """Metadata about a stored solution including its ID and block number."""
    id: SolutionId
    block: Optional[int] = None

class ChainManager:
    """Enhanced chain manager for storing and retrieving solution metadata."""
    
    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.wallet = wallet
        self.subnet_uid = subnet_uid
    
    def store_solution_metadata(self, hotkey: str, solution_id: SolutionId):
        """Stores solution metadata on the chain for a specific wallet."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")
        
        try:
            # Get current block number
            current_block = self.subtensor.get_current_block()
            solution_id.block_number = current_block
            
            self.subtensor.commit(
                self.wallet,
                self.subnet_uid,
                solution_id.to_compressed_str(),
            )

            logging.info(f"Attempted Submission to chain {hotkey}")
            return current_block
        except Exception as e:
            logging.error(f"Failed to store solution metadata: {str(e)}")
            raise
    
    def retrieve_solution_metadata(self, hotkey: str) -> Optional[SolutionMetadata]:
        """Retrieves solution metadata from the chain for a specific hotkey"""
        try:
            metadata = bt.core.extrinsics.serving.get_metadata(
                self.subtensor, 
                self.subnet_uid, 
                hotkey
            )
            if not metadata:
                return None
            
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            chain_str = bytes.fromhex(hex_data).decode()

            try:
                solution_id = SolutionId.from_compressed_str(chain_str)
                return SolutionMetadata(id=solution_id, block=metadata["block"])
            except Exception as e:
                logging.error(f"Failed to parse solution metadata for hotkey {hotkey}: {e}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to retrieve solution metadata: {str(e)}")
            return None
    
    def store_hf_repo(self, solution_id: SolutionId):
        """Stores solution metadata including repo and hash on the chain."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")
        
        try:
            # Get current block number
            current_block = self.subtensor.get_current_block()
            solution_id.block_number = current_block
            
            self.subtensor.commit(
                self.wallet,
                self.subnet_uid,
                solution_id.to_compressed_str(),
            )

            logging.info("Attempted Submission to chain ")

            return current_block
        except Exception as e:
            logging.error(f"Failed to store HF repo: {str(e)}")
            raise

    def store_raw_string(self, store_string: str):
        """Stores solution metadata including repo and hash on the chain."""
        if self.wallet is None:
            raise ValueError("No wallet available to write to the chain.")
        
        try:
            # Get current block number
            current_block = self.subtensor.get_current_block()
            # solution_id.block_number = current_block
            
            self.subtensor.commit(
                self.wallet,
                self.subnet_uid,
                store_string,
            )

            logging.info("Attempted Submission to chain ")

            return current_block
        except Exception as e:
            logging.error(f"Failed to store HF repo: {str(e)}")
            raise
    
    def retrieve_hf_repo(self, hotkey: str) -> Optional[str]:
        """Retrieves repository information from solution metadata."""
        metadata = self.retrieve_solution_metadata(hotkey)

        try:
            if metadata and metadata.id:
                return metadata.id.repo_name
        except:
            return None

    def get_submission_block(self, hotkey: str) -> Optional[int]:
        """Retrieves the block number when a solution was submitted."""
        metadata = self.retrieve_solution_metadata(hotkey)
        if metadata and metadata.id:
            return metadata.id.block_number
        return None