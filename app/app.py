from flask import Flask, render_template_string, request, jsonify
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

app = Flask(__name__)

# Charger le modele
MODEL_PATH = "./vulnerability_model_final"
if os.path.exists("./codebert_vuln_model"):
    MODEL_PATH = "./codebert_vuln_model"

print(f"Chargement du modele depuis {MODEL_PATH}...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Modele charge sur {device}")

THRESHOLD = 0.45

HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeBERT - Analyseur de Defauts</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            background: #0a0a1a;
            color: #e0e0f0;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .bg-glow {
            position: fixed;
            top: -200px;
            left: 50%;
            transform: translateX(-50%);
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        .bg-glow-2 {
            position: fixed;
            bottom: -300px;
            right: -200px;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(6, 182, 212, 0.1) 0%, transparent 70%);
            pointer-events: none;
            z-index: 0;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            position: relative;
            z-index: 1;
        }

        header {
            text-align: center;
            margin-bottom: 50px;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        }

        h1 {
            font-size: 2.2em;
            font-weight: 700;
            background: linear-gradient(135deg, #c7d2fe, #a5f3fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05em;
            margin-top: 8px;
            font-weight: 300;
        }

        .badge {
            display: inline-block;
            margin-top: 12px;
            padding: 6px 16px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 20px;
            font-size: 0.8em;
            color: #818cf8;
            font-weight: 500;
        }

        .card {
            background: rgba(15, 15, 35, 0.8);
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 20px;
            padding: 32px;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            margin-bottom: 24px;
            transition: border-color 0.3s ease;
        }

        .card:hover {
            border-color: rgba(99, 102, 241, 0.25);
        }

        .card-label {
            font-size: 0.85em;
            font-weight: 600;
            color: #818cf8;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 14px;
        }

        textarea {
            width: 100%;
            min-height: 200px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(99, 102, 241, 0.15);
            border-radius: 12px;
            padding: 18px;
            color: #e0e0f0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            line-height: 1.7;
            resize: vertical;
            outline: none;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }

        textarea::placeholder {
            color: #4b5563;
        }

        .btn-analyze {
            width: 100%;
            padding: 16px 32px;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 1.05em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
            letter-spacing: 0.3px;
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.25);
        }

        .btn-analyze:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(99, 102, 241, 0.35);
        }

        .btn-analyze:active {
            transform: translateY(0);
        }

        .btn-analyze:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .result-card {
            display: none;
            margin-top: 24px;
        }

        .result-card.show { display: block; }

        .result-safe {
            border-color: rgba(34, 197, 94, 0.3);
            background: rgba(34, 197, 94, 0.05);
        }

        .result-danger {
            border-color: rgba(239, 68, 68, 0.3);
            background: rgba(239, 68, 68, 0.05);
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
        }

        .result-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }

        .result-safe .result-icon {
            background: rgba(34, 197, 94, 0.15);
        }

        .result-danger .result-icon {
            background: rgba(239, 68, 68, 0.15);
        }

        .result-title {
            font-size: 1.5em;
            font-weight: 700;
        }

        .result-safe .result-title { color: #22c55e; }
        .result-danger .result-title { color: #ef4444; }

        .result-subtitle {
            color: #6b7280;
            font-size: 0.9em;
            margin-top: 2px;
        }

        .score-bar-container {
            margin-top: 16px;
        }

        .score-bar-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            margin-bottom: 8px;
            color: #9ca3af;
        }

        .score-bar {
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease;
        }

        .result-safe .score-bar-fill {
            background: linear-gradient(90deg, #22c55e, #4ade80);
        }

        .result-danger .score-bar-fill {
            background: linear-gradient(90deg, #ef4444, #f87171);
        }

        .examples {
            margin-top: 40px;
        }

        .examples h3 {
            font-size: 1.1em;
            color: #9ca3af;
            margin-bottom: 16px;
            font-weight: 500;
        }

        .example-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .example-btn {
            padding: 12px 16px;
            background: rgba(99, 102, 241, 0.06);
            border: 1px solid rgba(99, 102, 241, 0.12);
            border-radius: 10px;
            color: #a5b4fc;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .example-btn:hover {
            background: rgba(99, 102, 241, 0.12);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-1px);
        }

        .example-btn.danger { 
            border-color: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
        }
        .example-btn.danger:hover { 
            background: rgba(239, 68, 68, 0.08);
            border-color: rgba(239, 68, 68, 0.3);
        }

        footer {
            text-align: center;
            margin-top: 50px;
            padding: 24px;
            color: #4b5563;
            font-size: 0.85em;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 20px;
        }

        .stat {
            padding: 14px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            text-align: center;
        }

        .stat-value { font-size: 1.3em; font-weight: 700; color: #818cf8; }
        .stat-label { font-size: 0.75em; color: #6b7280; margin-top: 4px; }
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="bg-glow-2"></div>

    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">🛡️</div>
                <h1>CodeBERT Analyzer</h1>
            </div>
            <p class="subtitle">Detection intelligente de defauts dans le code Python</p>
            <span class="badge">Powered by Microsoft CodeBERT + Focal Loss</span>
        </header>

        <div class="card">
            <div class="card-label">Code Source</div>
            <textarea id="codeInput" placeholder="Collez votre code Python ici...

Exemples :
  def run(data): eval(data)
  import os; os.system(user_input)
  def add(a, b): return a + b"></textarea>
        </div>

        <button class="btn-analyze" id="btnAnalyze" onclick="analyser()">
            <span id="btnText">🔍 Analyser le code</span>
            <div class="spinner" id="spinner"></div>
        </button>

        <div class="result-card card" id="resultCard">
            <div class="result-header">
                <div class="result-icon" id="resultIcon"></div>
                <div>
                    <div class="result-title" id="resultTitle"></div>
                    <div class="result-subtitle" id="resultSubtitle"></div>
                </div>
            </div>
            <div class="score-bar-container">
                <div class="score-bar-label">
                    <span>Score de defaut</span>
                    <span id="scoreValue"></span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill" id="scoreBar"></div>
                </div>
            </div>
            <div class="stats" id="statsSection">
                <div class="stat">
                    <div class="stat-value" id="statScore">-</div>
                    <div class="stat-label">Score</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statConf">-</div>
                    <div class="stat-label">Confiance</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="statSeuil">45%</div>
                    <div class="stat-label">Seuil</div>
                </div>
            </div>
        </div>

        <div class="examples">
            <h3>Exemples de test</h3>
            <div class="example-grid">
                <button class="example-btn danger" onclick="setCode('def run(data): eval(data)')">eval(data) — exec dangereuse</button>
                <button class="example-btn" onclick="setCode('def add(a, b): return a + b')">add(a,b) — fonction normale</button>
                <button class="example-btn danger" onclick="setCode('import os; os.system(user_input)')">os.system — injection cmd</button>
                <button class="example-btn" onclick="setCode('def factorial(n): return 1 if n <= 1 else n * factorial(n-1)')">factorial — recursion normale</button>
                <button class="example-btn danger" onclick="setCode(`def query(name): db.execute('SELECT * FROM users WHERE name=' + name)`)">SQL injection</button>
                <button class="example-btn" onclick="setCode('import hashlib; hashlib.sha256(pwd.encode()).hexdigest()')">hashlib — hachage normal</button>
                <button class="example-btn danger" onclick="setCode('import pickle; data = pickle.loads(user_data)')">pickle.loads — deserialization</button>
                <button class="example-btn" onclick="setCode('class Calc:\\n  def add(self, a, b): return a + b')">class — code normal</button>
            </div>
        </div>

        <footer>
            CodeBERT Analyzer &mdash; Modele fine-tune sur le dataset Defectors (MSR 2023)<br>
            Architecture : RoBERTa + Focal Loss | Seuil de decision : 0.45
        </footer>
    </div>

    <script>
        function setCode(code) {
            document.getElementById('codeInput').value = code;
            analyser();
        }

        async function analyser() {
            const code = document.getElementById('codeInput').value;
            if (!code.trim()) return;

            const btn = document.getElementById('btnAnalyze');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');

            btn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'block';

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: code })
                });
                const data = await res.json();

                const card = document.getElementById('resultCard');
                const icon = document.getElementById('resultIcon');
                const title = document.getElementById('resultTitle');
                const subtitle = document.getElementById('resultSubtitle');
                const scoreValue = document.getElementById('scoreValue');
                const scoreBar = document.getElementById('scoreBar');
                const statScore = document.getElementById('statScore');
                const statConf = document.getElementById('statConf');

                card.classList.remove('result-safe', 'result-danger');

                if (data.defective) {
                    card.classList.add('result-danger');
                    icon.textContent = '⚠️';
                    title.textContent = 'CODE DEFECTUEUX';
                    subtitle.textContent = 'Ce code contient potentiellement un defaut ou une vulnerabilite';
                } else {
                    card.classList.add('result-safe');
                    icon.textContent = '✅';
                    title.textContent = 'CODE NORMAL';
                    subtitle.textContent = 'Ce code semble sain et sans defaut detecte';
                }

                const pct = (data.score * 100).toFixed(1);
                scoreValue.textContent = pct + '%';
                scoreBar.style.width = pct + '%';
                statScore.textContent = pct + '%';
                statConf.textContent = data.defective 
                    ? (data.score * 100).toFixed(0) + '%' 
                    : ((1 - data.score) * 100).toFixed(0) + '%';

                card.classList.add('show');
            } catch (e) {
                alert('Erreur: ' + e.message);
            }

            btn.disabled = false;
            btnText.style.display = 'inline';
            spinner.style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    code = data.get('code', '')
    
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
    
    score = probs[0][1].item()
    
    return jsonify({
        'score': score,
        'defective': score >= THRESHOLD,
        'label': 'DEFECTUEUX' if score >= THRESHOLD else 'NORMAL'
    })

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  CodeBERT Analyzer")
    print("  Ouvrez http://localhost:5000 dans le navigateur")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)
