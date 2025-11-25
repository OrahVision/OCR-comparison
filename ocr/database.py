"""
OCR Results Database

SQLite database for storing OCR results from multiple providers.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class OCRRecord:
    """A single OCR result record."""
    id: Optional[int]
    image_path: str
    provider: str
    text: str
    confidence: float
    word_count: int
    processing_time_ms: float
    created_at: str
    raw_response: Optional[str] = None


class OCRDatabase:
    """SQLite database for OCR results."""

    def __init__(self, db_path: str = "ocr_results.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Images table - tracks all processed images
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                folder TEXT NOT NULL,
                file_size INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # OCR results table - stores results from each provider
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                text TEXT,
                confidence REAL DEFAULT 0.0,
                word_count INTEGER DEFAULT 0,
                char_count INTEGER DEFAULT 0,
                processing_time_ms REAL DEFAULT 0.0,
                success INTEGER DEFAULT 1,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images(id),
                UNIQUE(image_id, provider)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_results_image_id
            ON ocr_results(image_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_results_provider
            ON ocr_results(provider)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_path
            ON images(path)
        """)

        self.conn.commit()

    def add_image(self, path: str) -> int:
        """
        Add an image to the database or get existing ID.

        Args:
            path: Path to the image file

        Returns:
            Image ID
        """
        cursor = self.conn.cursor()

        # Normalize path
        path = os.path.normpath(path)
        filename = os.path.basename(path)
        folder = os.path.dirname(path)
        file_size = os.path.getsize(path) if os.path.exists(path) else 0

        # Try to insert or get existing
        cursor.execute("""
            INSERT OR IGNORE INTO images (path, filename, folder, file_size)
            VALUES (?, ?, ?, ?)
        """, (path, filename, folder, file_size))

        cursor.execute("SELECT id FROM images WHERE path = ?", (path,))
        result = cursor.fetchone()
        self.conn.commit()

        return result['id']

    def add_ocr_result(
        self,
        image_id: int,
        provider: str,
        text: str,
        confidence: float = 0.0,
        word_count: int = 0,
        char_count: int = 0,
        processing_time_ms: float = 0.0,
        success: bool = True,
        error: str = None
    ) -> int:
        """
        Add or update an OCR result.

        Args:
            image_id: Image ID from images table
            provider: Provider name
            text: Extracted text
            confidence: Confidence score (0-1)
            word_count: Number of words
            char_count: Number of characters
            processing_time_ms: Processing time
            success: Whether OCR succeeded
            error: Error message if failed

        Returns:
            Result ID
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO ocr_results
            (image_id, provider, text, confidence, word_count, char_count,
             processing_time_ms, success, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            image_id, provider, text, confidence, word_count, char_count,
            processing_time_ms, 1 if success else 0, error
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_image_by_id(self, image_id: int) -> Optional[Dict]:
        """Get image info by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_image_by_path(self, path: str) -> Optional[Dict]:
        """Get image info by path."""
        cursor = self.conn.cursor()
        path = os.path.normpath(path)
        cursor.execute("SELECT * FROM images WHERE path = ?", (path,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_ocr_results_for_image(self, image_id: int) -> List[Dict]:
        """Get all OCR results for an image."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ocr_results
            WHERE image_id = ?
            ORDER BY provider
        """, (image_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_ocr_result(self, image_id: int, provider: str) -> Optional[Dict]:
        """Get OCR result for specific image and provider."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ocr_results
            WHERE image_id = ? AND provider = ?
        """, (image_id, provider))
        row = cursor.fetchone()
        return dict(row) if row else None

    def has_ocr_result(self, image_id: int, provider: str) -> bool:
        """Check if OCR result exists for image and provider."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 1 FROM ocr_results
            WHERE image_id = ? AND provider = ?
        """, (image_id, provider))
        return cursor.fetchone() is not None

    def get_all_ocr_results_for_image(self, image_id: int) -> List[Dict]:
        """
        Get all OCR results for a specific image from all providers.

        Args:
            image_id: Image ID

        Returns:
            List of OCR result records, one per provider
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ocr_results
            WHERE image_id = ?
            ORDER BY provider
        """, (image_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_images(self, folder_filter: str = None) -> List[Dict]:
        """
        Get all images, optionally filtered by folder.

        Args:
            folder_filter: Optional folder path to filter by

        Returns:
            List of image records
        """
        cursor = self.conn.cursor()

        if folder_filter:
            cursor.execute("""
                SELECT * FROM images
                WHERE folder LIKE ?
                ORDER BY folder, filename
            """, (f"%{folder_filter}%",))
        else:
            cursor.execute("""
                SELECT * FROM images
                ORDER BY folder, filename
            """)

        return [dict(row) for row in cursor.fetchall()]

    def get_images_with_provider_count(self) -> List[Dict]:
        """Get all images with count of OCR providers processed."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                i.*,
                COUNT(DISTINCT o.provider) as provider_count,
                GROUP_CONCAT(DISTINCT o.provider) as providers
            FROM images i
            LEFT JOIN ocr_results o ON i.id = o.image_id
            GROUP BY i.id
            ORDER BY i.folder, i.filename
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_providers(self) -> List[str]:
        """Get list of all providers with results."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT provider FROM ocr_results ORDER BY provider
        """)
        return [row['provider'] for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM images")
        image_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM ocr_results")
        result_count = cursor.fetchone()['count']

        cursor.execute("""
            SELECT provider, COUNT(*) as count, AVG(processing_time_ms) as avg_time
            FROM ocr_results
            GROUP BY provider
        """)
        provider_stats = {
            row['provider']: {
                'count': row['count'],
                'avg_time_ms': row['avg_time']
            }
            for row in cursor.fetchall()
        }

        return {
            'image_count': image_count,
            'result_count': result_count,
            'providers': provider_stats
        }

    def get_images_missing_provider(self, provider: str) -> List[Dict]:
        """Get images that don't have results for a specific provider."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT i.* FROM images i
            WHERE i.id NOT IN (
                SELECT image_id FROM ocr_results WHERE provider = ?
            )
            ORDER BY i.folder, i.filename
        """, (provider,))
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
def get_db(db_path: str = "ocr_results.db") -> OCRDatabase:
    """Get database instance."""
    return OCRDatabase(db_path)


if __name__ == "__main__":
    # Test database
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "ocr_results.db"

    with OCRDatabase(db_path) as db:
        stats = db.get_stats()
        print(f"Database: {db_path}")
        print(f"Images: {stats['image_count']}")
        print(f"Results: {stats['result_count']}")
        print("\nProvider stats:")
        for provider, pstats in stats['providers'].items():
            print(f"  {provider}: {pstats['count']} results, avg {pstats['avg_time_ms']:.0f}ms")
