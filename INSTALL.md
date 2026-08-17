# Installation & Setup Guide (Windows + PowerShell + VS Code)

This assumes the same setup you already use for your other projects: Windows, VS Code, PowerShell terminal. Everything below is copy-pasteable into a PowerShell terminal.

---

## 0. Prerequisites

| Tool | Needed for | Install from |
|---|---|---|
| **Python 3.10+** | Everything (data, ML, notebooks, Streamlit) | https://www.python.org/downloads/ — check "Add Python to PATH" during install |
| **VS Code** | Editing, running notebooks | You already have this |
| **VS Code "Jupyter" extension** | Running the .ipynb files | Install from VS Code Extensions panel (search "Jupyter", by Microsoft) |
| **MySQL Community Server** *(optional)* | Running the SQL analysis scripts | https://dev.mysql.com/downloads/mysql/ |
| **Power BI Desktop** *(optional, Windows only)* | Building the dashboard | Free from Microsoft Store or https://powerbi.microsoft.com/desktop/ |

Check Python is installed correctly:
```powershell
python --version
```
Should show `Python 3.10` or higher. If it says "not recognized," reinstall Python and make sure "Add to PATH" was checked.

---

## 1. Extract the project

Unzip `airtel-customer-churn.zip` into your projects folder:
```powershell
cd D:\MYPROJECTS
Expand-Archive -Path "$HOME\Downloads\airtel-customer-churn.zip" -DestinationPath .
cd D:\MYPROJECTS\airtel-customer-churn
```
(Adjust the source path if you saved the zip somewhere other than Downloads.)

---

## 2. Create a virtual environment and install packages

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**If you get an error about execution policy** ("running scripts is disabled on this system"), run this once, then retry the activate command:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Once activated, your prompt should show `(venv)` at the start. Now install everything:
```powershell
pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, plotly, joblib, jupyter, and mysql-connector-python. Takes a few minutes.

---

## 3. Verify the dataset

The dataset is already included at `data\airtel_enterprise_churn.csv` — you don't need to regenerate it. But if you ever want to (e.g. to get a different random sample size):
```powershell
cd src
python generate_dataset.py 25000 ..\data\airtel_enterprise_churn.csv
cd ..
```

---

## 4. Train the model

```powershell
cd src
python train_model.py
```
Takes 1-2 minutes. You'll see a model comparison table print out, and it saves `models\churn_model.pkl`. (The trained model is already included in the zip, so this step is optional unless you want to retrain from scratch.)

---

## 5. Generate churn predictions for every customer

```powershell
python prediction.py
cd ..
```
This creates `models\churn_predictions.csv` (used by the Streamlit dashboard and the SQL `churn_predictions` table) and prints the top 10 highest-priority at-risk customers.

---

## 6. Run the notebooks in VS Code

1. Open the project folder in VS Code: `code .` (from the project root, with venv still active)
2. Open `notebooks\01_EDA_Airtel_Churn.ipynb`
3. Click **"Select Kernel"** in the top-right → choose the `venv` Python interpreter (it should show as `Python 3.x.x ('venv')`)
4. Click **"Run All"** at the top of the notebook

Do the same for `02_Churn_Prediction_Model.ipynb`. Both should run top to bottom with no errors — I tested the full logic of both before packaging.

---

## 7. Run the Streamlit dashboard

```powershell
cd dashboard
streamlit run streamlit_app.py
```
This opens your browser automatically to `http://localhost:8501`. If it doesn't open automatically, click the URL PowerShell prints out. Use `Ctrl+C` in the terminal to stop it.

**If port 8501 is already in use:**
```powershell
streamlit run streamlit_app.py --server.port 8502
```

---

## 8. Set up MySQL (optional — only needed for the SQL analysis scripts)

1. Install MySQL Community Server (link above), remembering the root password you set during install.
2. Enable local file loading (needed for `LOAD DATA LOCAL INFILE`). Open **MySQL Command Line Client** or **MySQL Workbench** and run:
   ```sql
   SET GLOBAL local_infile = 1;
   ```
3. In `sql\00_schema_and_load.sql`, the `LOAD DATA LOCAL INFILE` paths use relative paths (`../data/...`). Either run the script from MySQL Workbench with the working directory set to the `sql\` folder, or replace the paths with the full Windows path, e.g.:
   ```sql
   LOAD DATA LOCAL INFILE 'D:/MYPROJECTS/airtel-customer-churn/data/airtel_enterprise_churn.csv'
   ```
   (Use forward slashes even on Windows — MySQL expects them.)
4. Run the two scripts in order, in MySQL Workbench (open file → Execute) or via command line:
   ```powershell
   mysql -u root -p --local-infile=1 < sql\00_schema_and_load.sql
   mysql -u root -p --local-infile=1 airtel_churn < sql\airtel_churn_analysis.sql
   ```
5. After running `prediction.py` (step 5), also load the predictions table:
   ```sql
   LOAD DATA LOCAL INFILE 'D:/MYPROJECTS/airtel-customer-churn/models/churn_predictions_sql.csv'
   INTO TABLE churn_predictions
   FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
   LINES TERMINATED BY '\n'
   IGNORE 1 ROWS;
   ```

---

## 9. Build the Power BI dashboard (optional)

1. Open Power BI Desktop
2. **Get Data → Text/CSV** → import `data\airtel_enterprise_churn.csv`, `sql\service_catalog.csv`, `sql\customer_services.csv`, `models\churn_predictions.csv`
3. Follow `reports\powerbi_dashboard_design.md` exactly — it has the relationships to draw, every DAX measure to paste in, and the layout for all 5 pages. Should take 1-2 hours end to end.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Reinstall Python, check "Add to PATH" |
| `pip install` fails on a package | Run `python -m pip install --upgrade pip` first, then retry |
| Notebook kernel doesn't show `venv` | In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → browse to `venv\Scripts\python.exe` |
| Streamlit shows a blank/error page | Make sure you ran `train_model.py` and `prediction.py` first — the app needs `models\churn_model.pkl` and `models\churn_predictions.csv` to exist |
| `ModuleNotFoundError` for any package | Make sure `(venv)` shows in your PowerShell prompt — you may have forgotten to activate it in a new terminal window |
| MySQL `LOAD DATA LOCAL INFILE` gives a permissions error | Add `--local-infile=1` to your mysql command, and confirm `SET GLOBAL local_infile = 1;` was run |
| Want to start fresh each new terminal session | Just re-run `.\venv\Scripts\Activate.ps1` from the project root — no need to reinstall packages |

---

## Quick reference — everything after first-time setup

Once installed, this is all you need for future sessions:
```powershell
cd D:\MYPROJECTS\airtel-customer-churn
.\venv\Scripts\Activate.ps1
cd dashboard
streamlit run streamlit_app.py
```
