from sqlalchemy import create_engine, inspect, text
import pandas as pd


class DatabaseLoader:
    """
    Connects to any SQLAlchemy-supported database and extracts
    schema information plus sample rows for embedding.
    """

    def __init__(self, connection_string: str):
        self._engine = create_engine(connection_string)

    def load(self, table_names: list[str] | None = None) -> list[dict]:
        inspector = inspect(self._engine)
        tables = table_names or inspector.get_table_names()
        results = []

        for table in tables:
            columns = inspector.get_columns(table)
            col_info = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in columns
            ]
            with self._engine.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 200"), conn)

            results.append({
                "table_name": table,
                "schema":     col_info,
                "dataframe":  df,
                "row_count":  len(df),
            })

        return results
