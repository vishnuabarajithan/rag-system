from pathlib import Path
import pandas as pd
import openpyxl


class SpreadsheetLoader:
    """
    Loads .xlsx / .xls / .ods files.
    Returns one dict per sheet with the full DataFrame and metadata.
    """

    def load(self, path: Path) -> list[dict]:
        ext = path.suffix.lower()
        engine = "openpyxl" if ext in (".xlsx", ".xlsm") else "xlrd"

        xl = pd.ExcelFile(str(path), engine=engine)
        sheets = []

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
