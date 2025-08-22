# dbconfig.py

import os
from datetime import datetime
from dotenv import load_dotenv
import pyodbc

# Load variables from .env file
load_dotenv()

SQLSERVER_SERVER = os.environ.get("SQLSERVER_SERVER", "")
SQLSERVER_DATABASE = os.environ.get("SQLSERVER_DATABASE", "")
SQLSERVER_USER = os.environ.get("SQLSERVER_USER", "")
SQLSERVER_PASSWORD = os.environ.get("SQLSERVER_PASSWORD", "")
SQLSERVER_DRIVER = os.environ.get("SQLSERVER_DRIVER", "")

def _build_conn_str() -> str:
    """
    Build and validate a SQL Server connection string from env vars.
    Raises a clear error if anything is missing.
    """
    required = {
        "SQLSERVER_SERVER": SQLSERVER_SERVER,
        "SQLSERVER_DATABASE": SQLSERVER_DATABASE,
        "SQLSERVER_USER": SQLSERVER_USER,
        "SQLSERVER_PASSWORD": SQLSERVER_PASSWORD,
        "SQLSERVER_DRIVER": SQLSERVER_DRIVER,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing SQL Server configuration for: {', '.join(missing)}. "
            "Set them in your environment or .env file."
        )

    driver = SQLSERVER_DRIVER
    if not driver.startswith("{"):
        driver = "{" + driver + "}"

    # Encrypt + TrustServerCertificate keeps compatibility in dev and most corp setups.
    return (
        f"DRIVER={driver};"
        f"SERVER={SQLSERVER_SERVER};"
        f"DATABASE={SQLSERVER_DATABASE};"
        f"UID={SQLSERVER_USER};"
        f"PWD={SQLSERVER_PASSWORD};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
    )

def _db_conn() -> pyodbc.Connection:
    """
    Open a new SQL Server connection. Uses autocommit so DDL/UPSERTs apply immediately.
    """
    conn_str = _build_conn_str()
    return pyodbc.connect(conn_str, autocommit=True)

def ensure_tables():
    """
    Create the S3 ingestion bookkeeping tables if they don't exist.
    """
    ddl_s3_ingest = r"""
    IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[s3_ingest]') AND type = N'U')
    BEGIN
        CREATE TABLE [dbo].[s3_ingest] (
            [id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            [bucket] NVARCHAR(255) NOT NULL,
            [s3_key] NVARCHAR(1024) NOT NULL,
            [result_id] NVARCHAR(64) NOT NULL,
            [processed_at] DATETIME2 NOT NULL
        );
        CREATE UNIQUE INDEX IX_s3_ingest_key ON [dbo].[s3_ingest]([bucket], [s3_key]);
    END
    """

    ddl_results = r"""
    IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[results]') AND type = N'U')
    BEGIN
        CREATE TABLE [dbo].[results] (
            [result_id] NVARCHAR(64) NOT NULL PRIMARY KEY,
            [filename]  NVARCHAR(512) NOT NULL,
            [s3_key]    NVARCHAR(1024) NULL,
            [created_at] DATETIME2 NOT NULL CONSTRAINT DF_results_created_at DEFAULT SYSUTCDATETIME(),
            [status] NVARCHAR(50) NOT NULL CONSTRAINT DF_results_status DEFAULT 'processed'
        );
        CREATE INDEX IX_results_created_at ON [dbo].[results]([created_at] DESC);
    END

    IF NOT EXISTS (SELECT * FROM sys.columns WHERE Name = N'status' AND Object_ID = Object_ID(N'dbo.results'))
    BEGIN
        ALTER TABLE [dbo].[results] ADD [status] NVARCHAR(50) NOT NULL CONSTRAINT DF_results_status DEFAULT 'processed';
    END
    """

    cn = _db_conn()
    cur = cn.cursor()
    try:
        cur.execute(ddl_s3_ingest)
        cur.execute(ddl_results)
    finally:
        try:
            cur.close()
        finally:
            cn.close()

def db_is_processed(bucket: str, s3_key: str) -> bool:
    """
    True if this S3 key was already processed (present in s3_ingest).
    """
    sql = "SELECT TOP (1) 1 FROM [dbo].[s3_ingest] WHERE [bucket]=? AND [s3_key]=?"
    cn = _db_conn()
    cur = cn.cursor()
    try:
        cur.execute(sql, (bucket, s3_key))
        return cur.fetchone() is not None
    finally:
        try:
            cur.close()
        finally:
            cn.close()

def db_mark_processed(bucket: str, s3_key: str, result_id: str):
    """
    Insert a row marking the given (bucket, s3_key) processed -> result_id.
    """
    sql = """
    INSERT INTO [dbo].[s3_ingest] ([bucket],[s3_key],[result_id],[processed_at])
    VALUES (?,?,?,?)
    """
    cn = _db_conn()
    cur = cn.cursor()
    try:
        cur.execute(sql, (bucket, s3_key, result_id, datetime.utcnow()))
    finally:
        try:
            cur.close()
        finally:
            cn.close()

def db_upsert_result(result_id: str, filename: str, s3_key: str | None):
    """
    Upsert into results table. If result_id exists, update filename/s3_key; else insert.
    The 'status' will be set to 'processed' by the database default.
    """
    cn = _db_conn()
    cur = cn.cursor()
    try:
        cur.execute(
            "UPDATE [dbo].[results] SET [filename]=?, [s3_key]=? WHERE [result_id]=?",
            (filename, s3_key, result_id)
        )
        if cur.rowcount == 0:
            cur.execute(
                "INSERT INTO [dbo].[results] ([result_id],[filename],[s3_key]) VALUES (?,?,?)",
                (result_id, filename, s3_key)
            )
    finally:
        try:
            cur.close()
        finally:
            cn.close()
