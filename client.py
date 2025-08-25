"""
Claude Code API client for managing sessions, jobs, and file operations.

This module provides a reusable client for interacting with the Claude Code API service,
handling session lifecycle, job execution, and result retrieval.
"""

import logging
import os
import time
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---

class FileInput(BaseModel):
    path: str = Field(..., description="Relative path of the file in the workspace.")
    content: str = Field(..., description="UTF-8 encoded content of the file.")

class SessionCreateRequest(BaseModel):
    files: List[FileInput]
    ttl_seconds: int = Field(3600, gt=0)
    env: Optional[Dict[str, str]] = Field(default=None, description="Environment variables to apply for this session")

class FileInfo(BaseModel):
    path: str
    size_bytes: int

class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime
    input_files: List[FileInfo]

class JobRunRequest(BaseModel):
    prompt: str
    timeout_s: int = Field(600, gt=0, le=3600)

class JobRunResponse(BaseModel):
    job_id: str
    session_id: str
    status: str = "queued"
    created_at: datetime

class OutputFileResult(BaseModel):
    path: str
    size_bytes: int
    sha256: str
    download_url: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    session_id: str
    status: str # "queued", "running", "succeeded", "failed", "timed_out"
    error: Optional[str] = None
    prompt: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output_files: Optional[List[OutputFileResult]] = None

class SessionResultsResponse(BaseModel):
    session_id: str
    latest_job_id: Optional[str] = None
    output_files: List[OutputFileResult]
    result_json: Optional[Dict[str, Any]] = None


class ClaudeCodeClient:
    """
    Client for interacting with the Claude Code API service.
    
    Handles session management, job execution, polling, and cleanup.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, cleanup_on_exit: bool = False, use_deepseek: bool = False):
        """
        Initialize the Claude Code API client.
        
        Args:
            api_key: API key for authentication. Defaults to CLAUDE_CODE_API_KEY env var.
            base_url: Base URL for the API. Defaults to CLAUDE_CODE_BASE_URL env var.
            use_deepseek: If True, configure sessions to use DeepSeek via Anthropic-compatible endpoint when possible.
        """
        api_key_value = api_key or os.getenv("CLAUDE_CODE_API_KEY")
        base_url_value = base_url or os.getenv("CLAUDE_CODE_BASE_URL")

        if not api_key_value or not base_url_value:
            raise ValueError("CLAUDE_CODE_API_KEY and CLAUDE_CODE_BASE_URL must be set")

        self.api_key: str = api_key_value
        self.base_url: str = base_url_value.rstrip("/")

        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.cleanup_on_exit = cleanup_on_exit
        self.session_id: Optional[str] = None
        self.use_deepseek: bool = use_deepseek
    
    def create_session(self, files: List[FileInput], ttl_seconds: int = 3600) -> str:
        """
        Create a new session with the provided files.
        
        Args:
            files: List of files to upload to the session
            ttl_seconds: Time-to-live for the session in seconds
            
        Returns:
            Session ID
            
        Raises:
            RuntimeError: If session creation fails
        """
        logger.info("Creating session on Claude Code API service...")
        
        env_vars: Dict[str, str] = {}

        # Optional DeepSeek via Anthropic-compatible endpoint
        # If DEEPSEEK_API_KEY is present, configure Claude Code to use DeepSeek
        use_deepseek = self.use_deepseek
        if use_deepseek:
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                logger.warning("DEEPSEEK_API_KEY is not set, falling back to Anthropic")
                use_deepseek = False
            else:
                env_vars.update({
                    # Allow override via env, otherwise default to DeepSeek base URL
                    "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),
                    # DeepSeek requires ANTHROPIC_AUTH_TOKEN
                    "ANTHROPIC_AUTH_TOKEN": deepseek_api_key,
                    # Default DeepSeek models; allow override if caller sets them
                    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "deepseek-chat"),
                    "ANTHROPIC_SMALL_FAST_MODEL": os.getenv("ANTHROPIC_SMALL_FAST_MODEL", "deepseek-chat"),
                })
        if not use_deepseek:
            # Standard Anthropic API key if available
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                env_vars["ANTHROPIC_API_KEY"] = anthropic_api_key

        # Use None if we have no env to send
        if not env_vars:
            env_vars = None

        session_request = SessionCreateRequest(
            files=files,
            ttl_seconds=ttl_seconds,
            env=env_vars
        )
        
        response = requests.post(
            f"{self.base_url}/session",
            headers=self.headers,
            json=session_request.model_dump()
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create session: {response.status_code} - {response.text}")
        
        session_create_response = SessionCreateResponse.model_validate(response.json())
        self.session_id = session_create_response.session_id
        logger.info(f"Created session: {self.session_id}")
        
        return self.session_id
    
    def run_job(self, prompt: str, timeout_s: int = 600) -> str:
        """
        Run a job in the current session.
        
        Args:
            prompt: The prompt to execute
            timeout_s: Timeout for the job in seconds
            
        Returns:
            Job ID
            
        Raises:
            RuntimeError: If job creation fails or no session exists
        """
        if not self.session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        
        job_request = JobRunRequest(
            prompt=prompt,
            timeout_s=timeout_s
        )
        
        response = requests.post(
            f"{self.base_url}/job",
            headers=self.headers,
            params={"session_id": self.session_id},
            json=job_request.model_dump()
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to start job: {response.status_code} - {response.text}")
        
        job_run_response = JobRunResponse.model_validate(response.json())
        job_id = job_run_response.job_id
        logger.info(f"Started job: {job_id}")
        
        return job_id
    
    def wait_for_job_completion(self, job_id: str, timeout_s: int = 600, poll_interval: int = 5) -> JobStatusResponse:
        """
        Poll for job completion.
        
        Args:
            job_id: ID of the job to monitor
            timeout_s: Maximum time to wait for completion
            poll_interval: Time between status checks in seconds
            
        Returns:
            Final job status response
            
        Raises:
            RuntimeError: If job fails, times out, or polling exceeds timeout
        """
        start_time = time.time()
        
        while True:
            response = requests.get(
                f"{self.base_url}/job/status",
                headers=self.headers,
                params={"job_id": job_id}
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get job status: {response.status_code} - {response.text}")
            
            job_status = JobStatusResponse.model_validate(response.json())
            status = job_status.status
            
            if status == "succeeded":
                logger.info("Job completed successfully")
                return job_status
            elif status == "failed":
                error_msg = job_status.error or "Unknown error"
                raise RuntimeError(f"Job failed: {error_msg}")
            elif status == "timed_out":
                raise RuntimeError("Job timed out on server")
            elif status in ["queued", "running"]:
                elapsed = time.time() - start_time
                if elapsed > timeout_s:
                    raise RuntimeError(f"Job timed out after {elapsed:.1f} seconds")
                logger.info(f"Job status: {status} (elapsed: {elapsed:.1f}s)")
                time.sleep(poll_interval)
            else:
                raise RuntimeError(f"Unknown job status: {status}")
    
    def get_session_results(self) -> SessionResultsResponse:
        """
        Get results from the current session.
        
        Returns:
            Session results including output files
            
        Raises:
            RuntimeError: If no session exists or request fails
        """
        if not self.session_id:
            raise RuntimeError("No active session")
        
        response = requests.get(
            f"{self.base_url}/session/results",
            headers=self.headers,
            params={"session_id": self.session_id}
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get results: {response.status_code} - {response.text}")
        
        session_results = SessionResultsResponse.model_validate(response.json())
        if session_results.result_json:
            # Best-effort labeling of provider vs workspace session IDs to avoid confusion in logs
            provider_session_id = None
            try:
                provider_session_id = session_results.result_json.get("session_id")  # type: ignore[attr-defined]
            except Exception:
                provider_session_id = None
            logger.info(
                "Response from Claude Code API service: provider_session_id=%s workspace_session_id=%s payload=%s",
                provider_session_id,
                self.session_id,
                session_results.result_json,
            )
        
        return session_results
    
    class TransientDownloadError(Exception):
        pass

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=0.75, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, TransientDownloadError))
    )
    def _download_text_with_retry(self, file_path: str) -> str:
        response = requests.get(
            f"{self.base_url}/download",
            headers=self.headers,
            params={
                "session_id": self.session_id,
                "file_path": file_path
            },
            timeout=60,
        )
        if 500 <= response.status_code < 600:
            raise self.TransientDownloadError(
                f"Failed to download output: {response.status_code} - {response.text}"
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download output: {response.status_code} - {response.text}"
            )
        return response.text

    def download_file(self, file_path: str) -> str:
        """
        Download a file from the current session.
        
        Args:
            file_path: Path of the file to download
            
        Returns:
            File contents as string
            
        Raises:
            RuntimeError: If no session exists or download fails
        """
        if not self.session_id:
            raise RuntimeError("No active session")

        logger.info(f"Downloading file {file_path} from session {self.session_id}")
        try:
            text = self._download_text_with_retry(file_path)
            logger.info(
                f"Downloaded file {file_path} from session {self.session_id} (size={len(text)} chars)"
            )
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to download output after retries: {e}")
    
    def delete_session(self) -> None:
        """
        Delete the current session.
        
        Handles cleanup gracefully with proper error logging.
        """
        if not self.session_id or not self.cleanup_on_exit:
            return
        
        try:
            response = requests.delete(
                f"{self.base_url}/session",
                headers=self.headers,
                params={"session_id": self.session_id},
                timeout=30
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Deleted session: {self.session_id}")
            elif response.status_code == 404:
                logger.info(f"Session {self.session_id} already deleted or expired")
            else:
                logger.warning(f"Failed to delete session: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error during session cleanup (session may auto-expire): {e}")
        except Exception as e:
            logger.warning(f"Unexpected error cleaning up session: {e}")
        finally:
            self.session_id = None
    
    def execute_job(
        self,
        prompt: str,
        inputs: Optional[dict[str, str]] = None,
        input_files: Optional[List[FileInput]] = None,
        output_dir: str = "out",
        timeout_s: int = 3600,
    ) -> str:
        """
        Execute a complete job with session management.
        
        This is a high-level method that handles the full workflow:
        1. Create session
        2. Populate prompt template with inputs
        2. Upload input files to session
        3. Run job
        4. Wait for completion
        5. Get results
        6. Download output files
        7. Clean up session
        
        Args:
            prompt: Prompt template for the job
            inputs: Inputs for the prompt template
            input_files: Input files to upload to the session
            output_dir: Directory to download output files to
            timeout_s: Job timeout in seconds
            
        Returns:
            Result JSON returned by the agent, if any
            
        Raises:
            RuntimeError: If any step in the process fails
        """        
        # Prepare input files
        files = input_files or []
        
        try:
            logger.info(
                "Creating session",
            )
            # Create session
            self.create_session(files, ttl_seconds=max(timeout_s * 2, 3600))
            logger.info("Session created session_id=%s", self.session_id)
            
            # Populate prompt template with inputs
            prompt = prompt.format(**inputs) if inputs else prompt

            # Run job
            job_id = self.run_job(prompt, timeout_s)
            logger.info("Job started job_id=%s", job_id)
            
            # Wait for completion
            self.wait_for_job_completion(job_id, timeout_s)
            logger.info("Job completed successfully job_id=%s", job_id)
            
            # Get results and verify output files exist
            results = self.get_session_results()
            output_files = results.output_files
            logger.info(
                "Retrieved session results files_count=%s",
            )

            if output_dir and output_files:
                # Make sure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                # Download the output files
                for file in output_files:
                    logger.info("Downloading output file", file.path)
                    content = self.download_file(file.path)
                    # Make sure any subdirectories in the path exist
                    os.makedirs(os.path.dirname(os.path.join(output_dir, file.path)), exist_ok=True)
                    # Write the content to the file
                    with open(os.path.join(output_dir, file.path), "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info("Downloaded output file", file.path)
                
                return results.result_json
        finally:
            # Always clean up
            logger.info("Cleaning up session session_id=%s", self.session_id)
            self.delete_session()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures session cleanup."""
        self.delete_session()


if __name__ == "__main__":
    # Example usage of the ClaudeCodeClient with asyncio semaphore for concurrency control
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    
    # Allow 3 concurrent jobs
    semaphore = asyncio.Semaphore(3)
    
    # Configure logging to see what's happening
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    client = ClaudeCodeClient(
        api_key=os.getenv("CLAUDE_CODE_API_KEY"),
        base_url=os.getenv("CLAUDE_CODE_BASE_URL"),
        cleanup_on_exit=True,
        use_deepseek=False,
    )
    
    # Execute job with template and inputs
    prompt_template = "Read {filename} and write a short story to {output_filename}"
    
    tasks = []
    for i in range(5):
        tasks.append(client.execute_job(
            prompt=prompt_template,
            inputs={"filename": f"topic_{i}.md", "output_filename": f"story_{i}.md"},
            input_files=FileInput(
                path=f"topic_{i}.md",
                content="The story should be about a cat."
            ),
            output_dir=f"output_{i}",
            timeout_s=120
        ))
    
    # Run tasks with concurrency control
    async def run_tasks():
        async with semaphore:
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    results = asyncio.run(run_tasks())
    for result in results:
        print(f"Job completed. Result: {result}")
