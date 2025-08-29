#!/usr/bin/env python3
"""
Database utilities for safe concurrent access to SQLite
"""

import sqlite3
import time
import threading
from contextlib import contextmanager
from typing import Optional, List, Tuple


class SafeSQLiteManager:
    """Thread-safe SQLite connection manager with retry logic"""
    
    def __init__(self, db_path: str, timeout: float = 30.0):
        self.db_path = db_path
        self.timeout = timeout
        self._local = threading.local()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            # Enable WAL mode for better concurrent access
            self._local.connection.execute('PRAGMA journal_mode=WAL')
            self._local.connection.execute('PRAGMA synchronous=NORMAL')
            self._local.connection.execute('PRAGMA cache_size=10000')
            self._local.connection.execute('PRAGMA temp_store=MEMORY')
        
        return self._local.connection
    
    @contextmanager
    def get_cursor(self, retries: int = 5):
        """Context manager for database operations with retry logic"""
        for attempt in range(retries):
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                yield cursor
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < retries - 1:
                    # Exponential backoff
                    wait_time = 0.1 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            except Exception as e:
                conn.rollback()
                raise
    
    def execute_with_retry(self, query: str, params: tuple = (), retries: int = 5) -> Optional[List[Tuple]]:
        """Execute a query with retry logic"""
        with self.get_cursor(retries=retries) as cursor:
            cursor.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            return None
    
    def execute_many_with_retry(self, query: str, param_list: List[tuple], retries: int = 5):
        """Execute many queries with retry logic"""
        with self.get_cursor(retries=retries) as cursor:
            cursor.executemany(query, param_list)


class SafeCrawlerDB:
    """Safe database operations for crawler"""
    
    def __init__(self, db_path: str):
        self.db_manager = SafeSQLiteManager(db_path)
    
    def get_next_urls(self, worker_id: int, domain_filter: str = None, limit: int = 10) -> List[Tuple]:
        """Get next URLs to crawl with proper locking"""
        where_clause = "status = 'pending' AND (attempts < 3 OR attempts IS NULL)"
        params = []
        
        if domain_filter:
            where_clause += " AND domain = ?"
            params.append(domain_filter)
        
        query = f"""
            SELECT id, url, domain, frequency 
            FROM urls 
            WHERE {where_clause}
            ORDER BY frequency DESC, id ASC
            LIMIT ?
        """
        params.append(limit)
        
        return self.db_manager.execute_with_retry(query, tuple(params)) or []
    
    def mark_processing(self, url_ids: List[int], worker_id: int):
        """Mark URLs as being processed (batch operation)"""
        if not url_ids:
            return
        
        placeholders = ','.join(['?'] * len(url_ids))
        query = f"""
            UPDATE urls 
            SET status = 'processing', 
                worker_id = ?,
                attempts = COALESCE(attempts, 0) + 1,
                last_attempt = ?
            WHERE id IN ({placeholders})
        """
        
        params = [worker_id, time.time()] + url_ids
        self.db_manager.execute_with_retry(query, tuple(params))
    
    def mark_completed(self, url_id: int, status: str, content_length: int = 0, 
                      file_path: str = None, error_message: str = None):
        """Mark URL as completed"""
        query = """
            UPDATE urls 
            SET status = ?,
                success_timestamp = ?,
                file_path = ?,
                content_length = ?,
                error_message = ?
            WHERE id = ?
        """
        
        success_time = time.time() if status == 'success' else None
        params = (status, success_time, file_path, content_length, error_message, url_id)
        
        self.db_manager.execute_with_retry(query, params)
    
    def update_domain_last_crawled(self, domain: str):
        """Update last crawled timestamp for domain"""
        query = """
            UPDATE domains 
            SET last_crawled = ? 
            WHERE domain = ?
        """
        
        self.db_manager.execute_with_retry(query, (time.time(), domain))
    
    def get_crawl_delay(self, domain: str) -> float:
        """Get appropriate crawl delay for domain"""
        query = "SELECT crawl_delay FROM domains WHERE domain = ?"
        result = self.db_manager.execute_with_retry(query, (domain,))
        
        return result[0][0] if result else 2.0
    
    def get_progress_stats(self) -> Tuple[int, int, int, int, int]:
        """Get overall progress statistics"""
        query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM urls
        """
        
        result = self.db_manager.execute_with_retry(query)
        return result[0] if result else (0, 0, 0, 0, 0)
    
    def get_worker_stats(self) -> dict:
        """Get per-worker statistics"""
        query = """
            SELECT worker_id, COUNT(*) 
            FROM urls 
            WHERE worker_id IS NOT NULL 
            GROUP BY worker_id
        """
        
        result = self.db_manager.execute_with_retry(query)
        return dict(result) if result else {}
    
    def get_domain_assignments(self, num_workers: int) -> dict:
        """Get balanced domain assignments for workers"""
        query = """
            SELECT domain, url_count, domain_type 
            FROM domains 
            WHERE url_count > 0
            ORDER BY url_count DESC
        """
        
        domains = self.db_manager.execute_with_retry(query) or []
        
        # Round-robin assignment
        assignments = {i: [] for i in range(1, num_workers + 1)}
        
        for i, (domain, count, dtype) in enumerate(domains):
            worker_id = (i % num_workers) + 1
            assignments[worker_id].append(domain)
        
        return assignments