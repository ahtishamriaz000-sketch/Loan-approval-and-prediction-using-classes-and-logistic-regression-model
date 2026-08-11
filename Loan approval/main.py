#================
#  AI LAB FYP
#==============


import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# =========================================================
# CLASS 1: DatasetLoader
# =========================================================
class DatasetLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        # Loan_ID ka ML mein use nahi
        df.drop(columns=["Loan_ID"], errors="ignore", inplace=True)
        return df


# =========================================================
# CLASS 2: DataPreprocessor (clean + encode)
# =========================================================
class DataPreprocessor:
    def __init__(self):
        self.encoders = {}

    def handle_missing(self, df):
        # categorical → mode
        cat = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
        for c in cat:
            df[c] = df[c].fillna(df[c].mode()[0])

        # numeric → median
        num = ["LoanAmount","Loan_Amount_Term","ApplicantIncome","CoapplicantIncome"]
        for c in num:
            df[c] = df[c].fillna(df[c].median())

        # credit history default good
        df["Credit_History"] = df["Credit_History"].fillna(1)
        return df

    def encode(self, df):
        cat = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
        for c in cat:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            self.encoders[c] = le

        # target
        df["Loan_Status"] = df["Loan_Status"].map({"Y":1, "N":0})
        return df

    def process(self, df):
        df = self.handle_missing(df)
        df = self.encode(df)
        return df


# =========================================================
# CLASS 3: ModelService (train + predict)
# =========================================================
class ModelService:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.columns = None
        self.accuracy = 0.0
        self.name = "Logistic Regression"

    def train(self, df):
        X = df.drop("Loan_Status", axis=1)
        y = df["Loan_Status"]

        self.columns = X.columns

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_tr, y_tr)

        # test accuracy 
        self.accuracy = round(self.model.score(X_te, y_te) * 100, 2)

    def predict(self, row):
        return self.model.predict([row])[0]


# =========================================================
# CLASS 4: FeatureBuilder (form → model input)
# =========================================================
class FeatureBuilder:
    def __init__(self, encoders, columns):
        self.encoders = encoders
        self.columns = columns

    def build(self, form):
        # mapping UI → dataset columns
        m = {
            "Gender": form["gender"],
            "Married": form["married"],
            "Dependents": form["dependents"],
            "Education": form["education"],
            "Self_Employed": form["self_employed"],
            "ApplicantIncome": float(form["income"]),
            "CoapplicantIncome": float(form["coincome"]),
            "LoanAmount": float(form["loan"]),
            "Loan_Amount_Term": float(form["term"]),
            "Credit_History": float(form["credit"]),
            "Property_Area": form["area"],
        }

        row = []
        for col in self.columns:
            val = m[col]

            # categorical encode
            if col in self.encoders:
                try:
                    val = self.encoders[col].transform([str(val)])[0]
                except:
                    val = 0  # unseen category safe fallback

            row.append(float(val))

        return row


# =========================================================
# CLASS 5: LoanApp (poora pipeline)
# =========================================================
class LoanApp:
    def __init__(self, csv_path):
        self.loader = DatasetLoader(csv_path)
        self.prep = DataPreprocessor()
        self.model = ModelService()

        self._train()

    def _train(self):
        df = self.loader.load()
        df = self.prep.process(df)
        self.model.train(df)

        # builder ko encoders + columns do
        self.builder = FeatureBuilder(self.prep.encoders, self.model.columns)

    def predict(self, form):
        row = self.builder.build(form)
        return self.model.predict(row)


# ================= INIT =================
system = LoanApp("train_u6lujuX_CVtuZ9i.csv")


# ================= FRONTEND (FULL UI) =================
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Loan Approval AI</title>

<style>
:root{
  --bg1:#0b1220; --bg2:#0f1b33; --card:rgba(255,255,255,0.06);
  --border:rgba(255,255,255,0.12); --txt:#e6edf7; --muted:#9fb0cc;
  --blue:#3b82f6; --green:#22c55e; --red:#ef4444; --gold:#f59e0b;
}
*{box-sizing:border-box}
body{
  margin:0; color:var(--txt); font-family:Segoe UI,system-ui;
  background: radial-gradient(800px 400px at -10% -10%, #1e3a8a33, transparent),
              radial-gradient(700px 400px at 110% 110%, #f59e0b22, transparent),
              linear-gradient(135deg,var(--bg1),var(--bg2));
  min-height:100vh; display:flex; align-items:center; justify-content:center; padding:20px;
}
.card{
  width:900px; max-width:95vw; padding:28px 26px; border-radius:24px;
  background:var(--card); border:1px solid var(--border); backdrop-filter: blur(18px);
}
.header{ text-align:center; margin-bottom:18px }
.header h1{ margin:6px 0; font-size:36px }
.header .badge{
  display:inline-block; padding:6px 14px; border-radius:999px;
  border:1px solid #f59e0b66; color:var(--gold); background:#f59e0b22; font-size:12px
}
.sub{ color:var(--muted); font-size:14px }

.grid{
  display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:14px;
}
.grid .span2{ grid-column: span 2 }
input,select{
  width:100%; padding:12px; border-radius:12px; border:1px solid var(--border);
  background:rgba(255,255,255,0.07); color:var(--txt)
}
button{
  width:100%; padding:14px; margin-top:14px; border-radius:14px;
  border:none; background:linear-gradient(135deg,#2563eb,#3b82f6);
  color:white; font-weight:700; cursor:pointer;
}

.info{
  display:flex; justify-content:space-between; gap:10px; margin-top:14px;
  border:1px solid var(--border); padding:12px; border-radius:12px;
}
.result{
  display:none; margin-top:14px; padding:16px; border-radius:14px; text-align:center;
}
.approved{ background:#22c55e22; border:1px solid #22c55e55 }
.rejected{ background:#ef444422; border:1px solid #ef444455 }
</style>
</head>

<body>
<div class="card">

<div class="header">
  <div class="badge">AI Powered</div>
  <h1>Loan Approval Predictor</h1>
  <div class="sub">Fill details — model will predict instantly</div>
</div>

<div class="grid">
<select id="gender"><option>Male</option><option>Female</option></select>
<select id="married"><option>Yes</option><option>No</option></select>
<select id="dependents"><option>0</option><option>1</option><option>2</option><option>3+</option></select>

<select id="education"><option>Graduate</option><option>Not Graduate</option></select>
<select id="self_employed"><option>No</option><option>Yes</option></select>
<select id="area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select>

<input type="number" id="income" placeholder="Applicant Income">
<input type="number" id="coincome" placeholder="Co Income" value="0">
<input type="number" id="loan" placeholder="Loan Amount">

<input class="span2" type="number" id="term" value="360">
<select id="credit">
<option value="1">Good Credit</option>
<option value="0">Bad Credit</option>
</select>
</div>

<button onclick="predict()">Predict</button>

<div class="result" id="resultBox">
  <h2 id="resText"></h2>
  <div id="resSub"></div>
</div>

<div class="info">
  <div>Model: <b id="modelName"></b></div>
  <div>Accuracy: <b id="acc"></b>%</div>
</div>

</div>

<script>
// load model info
fetch("/meta").then(r=>r.json()).then(d=>{
  document.getElementById("modelName").innerText = d.model;
  document.getElementById("acc").innerText = d.accuracy;
});

function predict(){
  let data = {
    gender: gender.value,
    married: married.value,
    dependents: dependents.value,
    education: education.value,
    self_employed: self_employed.value,
    income: income.value,
    coincome: coincome.value,
    loan: loan.value,
    term: term.value,
    credit: credit.value,
    area: area.value
  };

  fetch("/predict", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(data)
  })
  .then(r=>r.json())
  .then(res=>{
    let box = document.getElementById("resultBox");
    let t = document.getElementById("resText");
    let s = document.getElementById("resSub");

    box.style.display="block";

    if(res.approved){
      box.className="result approved";
      t.innerText="Loan Approved";
      s.innerText="High chance based on model prediction";
    } else {
      box.className="result rejected";
      t.innerText="Loan Rejected";
      s.innerText="Improve credit / income profile";
    }
  });
}
</script>

</body>
</html>
"""

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/meta")
def meta():
    # model name + accuracy UI par show
    return jsonify({
        "model": system.model.name,
        "accuracy": system.model.accuracy
    })

@app.route("/predict", methods=["POST"])
def predict():
    d = request.json
    pred = system.predict(d)
    return jsonify({"approved": bool(pred == 1)})

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)