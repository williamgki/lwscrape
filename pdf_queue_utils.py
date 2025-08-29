#!/usr/bin/env python3
"""
PDF Queue utilities for live PDF detection and processing
"""

import sqlite3
import threading
from urllib.parse import urlparse
from typing import Optional
from db_utils import SafeSQLiteManager

class PDFQueueManager:
    """Manages PDF detection and queueing"""
    
    def __init__(self, db_path: str):
        self.db_manager = SafeSQLiteManager(db_path)
    
    def add_pdf_to_queue(self, url: str, content_type: str) -> bool:
        """Add a detected PDF to the processing queue"""
        try:
            if not self.is_pdf_content_type(content_type):
                return False
                
            domain = urlparse(url).netloc
            
            with self.db_manager.get_cursor() as cursor:
                # Insert or update frequency if already exists
                cursor.execute('''
                    INSERT INTO pdf_queue (url, domain, frequency, status)
                    VALUES (?, ?, 1, 'pending')
                    ON CONFLICT(url) DO UPDATE SET
                        frequency = frequency + 1,
                        added_timestamp = CURRENT_TIMESTAMP
                ''', (url, domain))
                
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error adding PDF to queue: {e}")
            return False
    
    def is_pdf_content_type(self, content_type: str) -> bool:
        """Check if content type indicates a PDF"""
        if not content_type:
            return False
        content_type = content_type.lower()
        return ('pdf' in content_type or 
                content_type == 'application/pdf' or
                'application/pdf' in content_type)
    
    def get_pending_pdfs(self, limit: int = 50) -> list:
        """Get pending PDFs from queue, ordered by frequency"""
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute('''
                    SELECT url, domain, frequency 
                    FROM pdf_queue 
                    WHERE status = 'pending'
                    ORDER BY frequency DESC, added_timestamp ASC
                    LIMIT ?
                ''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            print(f"Error getting pending PDFs: {e}")
            return []
    
    def mark_pdf_processed(self, url: str, success: bool, file_path: Optional[str] = None, 
                          error_message: Optional[str] = None, content_length: int = 0):
        """Mark PDF as processed in queue"""
        try:
            status = 'success' if success else 'failed'
            with self.db_manager.get_cursor() as cursor:
                cursor.execute('''
                    UPDATE pdf_queue 
                    SET status = ?, processed_timestamp = CURRENT_TIMESTAMP,
                        file_path = ?, error_message = ?, content_length = ?
                    WHERE url = ?
                ''', (status, file_path, error_message, content_length, url))
        except Exception as e:
            print(f"Error marking PDF as processed: {e}")
    
    def get_queue_stats(self) -> dict:
        """Get PDF queue statistics"""
        try:
            with self.db_manager.get_cursor() as cursor:
                cursor.execute('''
                    SELECT status, COUNT(*) 
                    FROM pdf_queue 
                    GROUP BY status
                ''')
                stats = dict(cursor.fetchall())
                
                # Add total
                stats['total'] = sum(stats.values())
                return stats
        except Exception as e:
            print(f"Error getting queue stats: {e}")
            return {'total': 0, 'pending': 0, 'success': 0, 'failed': 0}