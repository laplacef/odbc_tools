# Pipeline Tools

This repository contains a Python module for connecting to SQL databases and some useful utility functions.

## SQLConnector

The SQLConnector class establishes connections to SQL Server, MySQL, and PostgreSQL databases. It provides methods to create an engine, create a new database if it doesn't exist, create a connection, and inspect tables and their respective columns.

### Methods

- **create_engine_mssql(database: str) -> object:** Create engine to MSSQL Server Database
- **create_engine_mysql(database: str, pw: str) -> object:** Create engine to MySQL Server Database
- **create_engine_postgresql(database: str) -> object:** Create engine to PostgreSQL Server Database
- **create_new_database(engine) -> None:** Creates database if it doesn't exist
- **create_connection(engine) -> object:** Returns SqlAlchemy Connection object if connection to the database is successful
- **inspect_tables(conn):** Inspect database tables and respective column names by loading them into a dictionary

### Usage

To use the SQLConnector class, create an instance and call the appropriate methods to connect to your desired database. For example, to create an engine for a PostgreSQL database:

``` python
connector = SQLConnector()
engine = connector.create_engine_postgresql("my_database")
```
