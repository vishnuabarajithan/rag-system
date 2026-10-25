from pathlib import Path
import pandas as pd
import openpyxl


class SpreadsheetLoader:
    """
    Loads .xlsx / .xls / .ods / .csv files.
    Returns one dict per sheet with the full DataFrame and metadata.
    """

    def load(self, path: Path, name: str | None = None) -> list[dict]:
        """`name`: display name for a CSV's single "sheet" (defaults to the
        file stem — callers ingesting a temp copy pass the original name)."""
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(str(path), encoding_errors="replace")
            df = df.dropna(how="all").fillna("")
            return [{
                "sheet_name": name or path.stem,
                "columns":    list(df.columns),
                "dataframe":  df,
                "row_count":  len(df),
            }]

        engine = "openpyxl" if ext in (".xlsx", ".xlsm") else "xlrd"

        sheets = []
        # Context manager closes the file handle — on Windows an open handle
        # makes the caller's temp-file cleanup fail (WinError 32).
        with pd.ExcelFile(str(path), engine=engine) as xl:
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                df = df.dropna(how="all").fillna("")
                sheets.append({
                    "sheet_name": sheet_name,
                    "columns":    list(df.columns),
                    "dataframe":  df,
                    "row_count":  len(df),
                })

        return sheets
