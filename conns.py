import os

from sqlalchemy import create_engine, inspect
from sqlalchemy_utils import database_exists, create_database


class SQLConnector:
    """Establish connection to SQL Servers."""

    def __init__(self):
        """Initialize attributes."""

        pass

    def create_engine_mssql(self, database) -> object:
        """
        Create engine to MSSQL Server Database.
        
        Parameters
        ----------
        database : str
            The name of the database required for connection
        """

        host = os.environ["COMPUTERNAME"]
        url = f"mssql+pyodbc://@{host}\SQLEXPRESS/{database}?Trusted_Connection=yes&driver=SQL Server"

        try:
            engine = create_engine(url)
        except:
            print("Could not create engine object.")
        else:
            print(f"Engine created.")
            return engine

    def create_engine_mysql(self, database, pw="") -> object:
        """
        Create engine to MySQL Server Database.
        
        Parameters
        ----------
        database : str
            The name of the database required for connection
        """

        host = "root"
        url = f"mysql+pymysql://{host}:{pw}@localhost:3306/{database}"

        try:
            engine = create_engine(url)
        except:
            print("Could not create engine object.")
        else:
            print(f"Engine created.")
            return engine

    def create_engine_postgresql(self, database) -> object:
        """
        Create engine to PostgreSQL Server Database.
        
        Parameters
        ----------
        database : str
            The name of the database required for connection
        """

        host = os.environ["COMPUTERNAME"]
        url = f"postgresql+psycopg2://{host}:tiger@localhost/{database}"

        try:
            engine = create_engine(url)
        except:
            print("Could not create engine object.")
        else:
            print(f"Engine created.")
            return engine

    def create_new_database(self, engine) -> None:
        """Creates database if does not exist."""

        if not database_exists(engine.url):
            create_database(engine.url)
            print(f"Created new database at: {engine.url}")
        else:
            print(f"Database already exists. To change Database, update engine URL.")

    def create_connection(self, engine) -> object:
        """Returns SqlAlchemy Connection object if connection to database is successful."""

        try:
            self.create_new_database(engine)
            conn = engine.connect()
        except:
            print(f"Engine could not connect to database. Status: FAIL")
        else:
            print(f"Connection to database created. Status: SUCCESS")
            return conn

    def inspect_tables(self, conn):
        """Inspect database tables and respective column names by loading them into dictionary."""

        inspector = inspect(conn)

        tables = {}
        for table in inspector.get_table_names():
            tables[table] = []
            for col in inspector.get_columns(table):
                tables[table].append(col["name"])

        return tables
