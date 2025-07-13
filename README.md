### Projektbeskrivelse og formål.

Den primære motivation bag dette projekt var at få praktisk erfaring med MLFlow for at opnå en bedre forståelse af MLOps og praktisk erfaring med MLFlow API’et generelt.

Min motivation var også at få praktisk erfaring med Docker og containerisering, da dette er en eftertragtet kompetence hos arbejdsgivere.

Derfor vil ML-modelarkitekturen ikke være særlig kompleks i starten, da jeg fokuserer på simple anvendelser af MLFlow og Docker. Men med tiden vil jeg udvikle arkitekturen til at blive mere kompleks.

### Installation og opsætning (fx miljø, dependencies).

Tjek blandt andet filen requirements.txt

### Instruktioner til hvordan man kører koden eller notebooks.

**nedenstående skal køres i powershell, for at få containeren op at køre med ML-modellen.**

**bygger et image:** 

docker build -t xgb-mlflow .

**(alternativt) bygger image uden cache:**

docker build --no-cache -t xgb-mlflow .

**kører containeren**

docker run --rm `
  -v C:\Job_og_eksamensbevis\Github\projekter\ML_ensemble_modeller_projekt-main\mlruns:/app/mlruns `
  -v C:\Job_og_eksamensbevis\Github\projekter\ML_ensemble_modeller_projekt-main\creditcard.csv:/app/creditcard.csv `
  -e MLFLOW_TRACKING_URI="file:///app/mlruns" `
  xgb-mlflow

**starter mlflow ui (indsæt egen sti til mlruns-mappen. Jeg ved godt dette ikke er best practice):**

mlflow ui --backend-store-uri file:///C:/Job_og_eksamensbevis/Github/projekter/ML_ensemble_modeller_projekt-main/mlruns


### Forklaring af datasæt og kilde.

**datasættet kan hentes på kaggle:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Resultater og konklusioner.
