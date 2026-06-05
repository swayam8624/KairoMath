import os

cpp_file_path = "/Users/swayamsingal/Desktop/Programming/Kairo/Foundation/KairoMath/tests/VisualTests.cpp"

with open(cpp_file_path, "r") as f:
    content = f.read()

# Locate start and end of GenerateHTMLVisualizer
start_token = "void GenerateHTMLVisualizer(const std::string& filepath)"
end_token = "// Function to animate a 3D wireframe cube rotating in the console terminal"

start_idx = content.find(start_token)
end_idx = content.find(end_token)

if start_idx == -1 or end_idx == -1:
    print(f"Error: Start or End token not found. Start: {start_idx}, End: {end_idx}")
    exit(1)

# The new implementation
new_html_generator = """void GenerateHTMLVisualizer(const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to generate visual_tests.html\\n";
        return;
    }

    file << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KairoMath Interactive Visual Lab</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0B0F19;
            --card-bg: #161B26;
            --border-color: #2D3748;
            --text-main: #E2E8F0;
            --text-muted: #718096;
            
            --vector-a: #FF6B6B;
            --vector-b: #4ECDC4;
            --project: #FFE66D;
            --normal: #A0AEC0;
            --refract: #9B5DE5;
            --reflect: #F15BB5;
            --accent: #4ECDC4;
            --danger: #E53E3E;
            --success: #38A169;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem;
        }

        header {
            max-width: 1200px;
            margin: 0 auto 2rem auto;
            text-align: center;
        }

        h1 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFF 0%, #4ECDC4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.2rem;
            font-weight: 300;
        }

        .status-container {
            display: inline-flex;
            align-items: center;
            background: rgba(22, 27, 38, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.5rem 1rem;
            border-radius: 30px;
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
            background-color: var(--danger);
            box-shadow: 0 0 8px var(--danger);
            transition: all 0.3s ease;
        }

        .status-dot.online {
            background-color: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        .tabs {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.75rem;
            max-width: 1200px;
            margin: 0 auto 2.5rem auto;
        }

        .tab-btn {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .tab-btn:hover {
            border-color: #4ECDC4;
            color: var(--text-main);
        }

        .tab-btn.active {
            color: #FFF;
            border-color: #4ECDC4;
            background-color: rgba(78, 205, 196, 0.1);
            box-shadow: 0 0 12px rgba(78, 205, 196, 0.25);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(8px);
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }

        @media (max-width: 900px) {
            .card {
                grid-template-columns: 1fr;
            }
        }

        .card-header {
            grid-column: 1 / -1;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 1.8rem;
            font-weight: 600;
            color: #FFF;
        }

        .visual-area {
            position: relative;
            background-color: #0F131C;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            min-height: 420px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .canvas-container {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        canvas, svg {
            background-color: transparent;
        }

        .controls {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .control-group {
            margin-bottom: 1.5rem;
        }

        .control-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 1rem;
            color: var(--text-main);
        }

        .control-value {
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent);
        }

        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #2D3748;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
            cursor: pointer;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #4ECDC4;
            transition: transform 0.1s ease;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        .equation-block {
            background-color: #0F131C;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            margin-top: 1.5rem;
            overflow-x: auto;
        }

        .eq-line {
            margin-bottom: 0.5rem;
        }

        .dot-a { color: var(--vector-a); }
        .dot-b { color: var(--vector-b); }
        .dot-proj { color: var(--project); }
        .dot-refract { color: var(--refract); }
        .dot-reflect { color: var(--reflect); }

        .badge {
            background-color: #2D3748;
            color: var(--text-main);
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: 0.5rem;
            display: inline-block;
        }

        .alert-tir {
            background-color: rgba(241, 91, 181, 0.15);
            border: 1px solid var(--reflect);
            color: #FFF;
            padding: 0.75rem;
            border-radius: 8px;
            margin-top: 1rem;
            display: none;
            text-align: center;
            font-weight: 600;
        }

        .matrix-grid {
            display: inline-grid;
            gap: 6px;
            background-color: #0F131C;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .matrix-cell {
            width: 60px;
            height: 38px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: #FFF;
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            outline: none;
        }

        .matrix-cell:focus {
            border-color: var(--accent);
            box-shadow: 0 0 5px rgba(78, 205, 196, 0.4);
        }

        .solution-vector {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-left: 10px;
            border-left: 2px solid var(--accent);
            padding-left: 12px;
        }

        .output-matrix-container {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.5rem;
            margin-top: 1.5rem;
        }

        .output-matrix {
            display: inline-grid;
            gap: 4px;
            border-left: 2px solid var(--text-muted);
            border-right: 2px solid var(--text-muted);
            padding: 0 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        .output-matrix-cell {
            width: 65px;
            text-align: center;
            padding: 4px 0;
        }

        .solve-btn {
            background-color: var(--accent);
            color: var(--bg-dark);
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 800;
            cursor: pointer;
            font-family: 'Outfit', sans-serif;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
            margin-top: 1rem;
        }

        .solve-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(78, 205, 196, 0.4);
        }

        .error-message {
            background-color: rgba(229, 62, 62, 0.15);
            border: 1px solid var(--danger);
            color: #FFAAAA;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            display: none;
        }

        .preset-select {
            background-color: #0F131C;
            color: var(--text-main);
            border: 1px solid var(--border-color);
            padding: 0.5rem;
            border-radius: 6px;
            font-family: 'Outfit', sans-serif;
            outline: none;
            cursor: pointer;
        }

        .pca-canvas-elem {
            border-radius: 8px;
            cursor: crosshair;
            background-color: #0F131C;
        }

        .pca-info {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            background: #1A202C;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }
    </style>
</head>
<body>

    <header>
        <h1>KairoMath Visualizer</h1>
        <div class="subtitle">Interactive Verification Dashboard linked directly to the C++ Engine</div>
        <div class="status-container">
            <div id="status-dot" class="status-dot"></div>
            <span id="status-text">C++ Engine: Offline (JS Fallback)</span>
        </div>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-vectors')">Vector Operations</button>
        <button class="tab-btn" onclick="switchTab('tab-solve')">Linear Systems</button>
        <button class="tab-btn" onclick="switchTab('tab-decompose')">Matrix Decompositions</button>
        <button class="tab-btn" onclick="switchTab('tab-eigen-svd')">Eigenvalues & SVD</button>
        <button class="tab-btn" onclick="switchTab('tab-pca-reg')">Interactive PCA & Regression</button>
        <button class="tab-btn" onclick="switchTab('tab-transform3d')">3D Transform Cube</button>
    </div>

    <div class="container">
        
        <!-- Vectors Tab -->
        <div id="tab-vectors" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Vector Operations Lab</span>
                    <span class="badge">Projection, Reflection & Refraction</span>
                </div>
                
                <div class="visual-area">
                    <svg id="vector-svg" width="450" height="420" viewBox="-225 -210 450 420" style="width:100%; height:100%;">
                        <line x1="-225" y1="0" x2="225" y2="0" stroke="#1E2433" stroke-width="1" />
                        <line x1="0" y1="-210" x2="0" y2="210" stroke="#1E2433" stroke-width="1" />
                        <circle cx="0" cy="0" r="4" fill="#FFF" />
                        
                        <line id="vec-axis" x1="-225" y1="0" x2="225" y2="0" stroke="#2D3748" stroke-dasharray="4,4" stroke-width="1.5" />
                        <line id="vec-b" x1="0" y1="0" x2="120" y2="40" stroke="var(--vector-b)" stroke-width="3.5" marker-end="url(#arr-b)" />
                        <line id="vec-a" x1="0" y1="0" x2="60" y2="100" stroke="var(--vector-a)" stroke-width="3.5" marker-end="url(#arr-a)" />
                        <line id="vec-proj" x1="0" y1="0" x2="0" y2="0" stroke="var(--project)" stroke-width="4" marker-end="url(#arr-proj)" />
                        <line id="vec-helper" x1="0" y1="0" x2="0" y2="0" stroke="#718096" stroke-dasharray="3,3" stroke-width="1.5" />
                        
                        <line id="normal-line" x1="0" y1="-200" x2="0" y2="200" stroke="var(--normal)" stroke-width="1.5" stroke-dasharray="5,5" style="display:none;" />
                        <line id="refract-boundary" x1="-225" y1="0" x2="225" y2="0" stroke="#2B6CB0" stroke-width="2" style="display:none;" />
                        <rect id="medium2-rect" x="-225" y="0" width="450" height="210" fill="rgba(43, 108, 176, 0.15)" style="display:none;" />
                        
                        <line id="ray-inc" x1="0" y1="0" x2="0" y2="0" stroke="var(--vector-a)" stroke-width="3" style="display:none;" />
                        <line id="ray-refl" x1="0" y1="0" x2="0" y2="0" stroke="var(--reflect)" stroke-width="3" style="display:none;" />
                        <line id="ray-refr" x1="0" y1="0" x2="0" y2="0" stroke="var(--refract)" stroke-width="3" style="display:none;" />
                        
                        <defs>
                            <marker id="arr-a" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="var(--vector-a)" /></marker>
                            <marker id="arr-b" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="var(--vector-b)" /></marker>
                            <marker id="arr-proj" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="var(--project)" /></marker>
                        </defs>
                    </svg>
                </div>
                
                <div class="controls">
                    <div>
                        <div style="display:flex; gap:0.5rem; margin-bottom:1.5rem;">
                            <button id="btn-mode-proj" class="tab-btn active" onclick="setVectorLabMode('projection')" style="padding:0.4rem 1rem; font-size:0.9rem;">Projection Mode</button>
                            <button id="btn-mode-refr" class="tab-btn" onclick="setVectorLabMode('refraction')" style="padding:0.4rem 1rem; font-size:0.9rem;">Reflection/Refraction</button>
                        </div>
                        <div id="ctrl-projection">
                            <div class="control-group">
                                <div class="control-label"><span>Vector A Angle</span><span class="control-value" id="val-vec-a-ang">60°</span></div>
                                <input type="range" id="slider-vec-a-ang" min="0" max="360" value="60" oninput="updateVectorLab()">
                            </div>
                            <div class="control-group">
                                <div class="control-label"><span>Vector A Magnitude</span><span class="control-value" id="val-vec-a-mag">120</span></div>
                                <input type="range" id="slider-vec-a-mag" min="10" max="180" value="120" oninput="updateVectorLab()">
                            </div>
                            <div class="control-group">
                                <div class="control-label"><span>Vector B Angle</span><span class="control-value" id="val-vec-b-ang">15°</span></div>
                                <input type="range" id="slider-vec-b-ang" min="0" max="360" value="15" oninput="updateVectorLab()">
                            </div>
                            <div class="control-group">
                                <div class="control-label"><span>Vector B Magnitude</span><span class="control-value" id="val-vec-b-mag">150</span></div>
                                <input type="range" id="slider-vec-b-mag" min="0" max="180" value="150" oninput="updateVectorLab()">
                            </div>
                        </div>
                        <div id="ctrl-refraction" style="display:none;">
                            <div class="control-group">
                                <div class="control-label"><span>Incident Ray Angle</span><span class="control-value" id="val-ray-ang">45°</span></div>
                                <input type="range" id="slider-ray-ang" min="0" max="89" value="45" oninput="updateVectorLab()">
                            </div>
                            <div class="control-group">
                                <div class="control-label"><span>Medium 1 Index (n₁)</span><span class="control-value" id="val-ray-n1">1.0</span></div>
                                <input type="range" id="slider-ray-n1" min="10" max="30" value="10" oninput="updateVectorLab()">
                            </div>
                            <div class="control-group">
                                <div class="control-label"><span>Medium 2 Index (n₂)</span><span class="control-value" id="val-ray-n2">1.5</span></div>
                                <input type="range" id="slider-ray-n2" min="10" max="30" value="15" oninput="updateVectorLab()">
                            </div>
                            <div class="alert-tir" id="tir-alert">TOTAL INTERNAL REFLECTION</div>
                        </div>
                    </div>
                    <div class="equation-block" id="vector-equation-block"></div>
                </div>
            </div>
        </div>

        <!-- Linear Solve Tab -->
        <div id="tab-solve" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Linear Solver Lab ($A x = b$)</span>
                    <div>
                        <select id="solve-preset" class="preset-select" onchange="applySolvePreset()">
                            <option value="unique">Unique Solution</option>
                            <option value="singular">Singular Matrix</option>
                            <option value="hilbert">Hilbert Matrix</option>
                        </select>
                        <select id="solve-dim" class="preset-select" style="margin-left:0.5rem;" onchange="recreateSolveGrid()">
                            <option value="2">2 x 2</option>
                            <option value="3" selected>3 x 3</option>
                            <option value="4">4 x 4</option>
                        </select>
                    </div>
                </div>
                <div>
                    <h3>Input Matrix A & Vector b</h3>
                    <div style="display:flex; align-items:center; flex-wrap:wrap; gap:1.5rem; margin-top:1rem;">
                        <div id="solve-matrix-container"></div>
                        <div style="font-size:1.5rem; font-family:'JetBrains Mono'; font-weight:bold;">x</div>
                        <div style="font-size:1.5rem; font-family:'JetBrains Mono'; font-weight:bold;">=</div>
                        <div id="solve-b-container"></div>
                    </div>
                    <button class="solve-btn" onclick="executeLinearSolve()">Solve via C++ Engine</button>
                    <div class="error-message" id="solve-error-box"></div>
                </div>
                <div class="controls">
                    <h3>Solution Results</h3>
                    <div class="equation-block">
                        <div style="font-weight:600; color:var(--accent);">Solution vector x:</div>
                        <div id="solve-x-result" class="solution-vector" style="margin-top:0.5rem; margin-bottom:1rem;"></div>
                        <div style="font-weight:600; color:var(--text-muted); border-top:1px solid var(--border-color); padding-top:0.5rem;">REF Matrix:</div>
                        <div id="solve-ref-matrix" class="output-matrix-container" style="margin-bottom:1rem;"></div>
                        <div style="font-weight:600; color:var(--text-muted); border-top:1px solid var(--border-color); padding-top:0.5rem;">RREF Matrix:</div>
                        <div id="solve-rref-matrix" class="output-matrix-container"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Decomposition Tab -->
        <div id="tab-decompose" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Matrix Decomposition Lab</span>
                    <div>
                        <select id="decomp-method" class="preset-select" onchange="executeDecomposition()">
                            <option value="lu">LU (A = L * U)</option>
                            <option value="lup" selected>LUP (P * A = L * U)</option>
                            <option value="qr">QR (A = Q * R)</option>
                            <option value="cholesky">Cholesky (A = L * Lᵀ)</option>
                            <option value="ldlt">LDLT (A = L * D * Lᵀ)</option>
                        </select>
                        <select id="decomp-dim" class="preset-select" style="margin-left:0.5rem;" onchange="recreateDecompGrid()">
                            <option value="2">2 x 2</option>
                            <option value="3" selected>3 x 3</option>
                            <option value="4">4 x 4</option>
                        </select>
                    </div>
                </div>
                <div>
                    <h3>Input Matrix A</h3>
                    <div id="decomp-matrix-container" style="margin-top:1rem;"></div>
                    <button class="solve-btn" onclick="executeDecomposition()">Factorize Matrix</button>
                    <div class="error-message" id="decomp-error-box"></div>
                </div>
                <div class="controls">
                    <h3>Decomposition Factors</h3>
                    <div class="equation-block" id="decomp-result-box"></div>
                </div>
            </div>
        </div>

        <!-- Eigen & SVD Tab -->
        <div id="tab-eigen-svd" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Eigenvalues & SVD</span>
                    <select id="eigen-dim" class="preset-select" onchange="recreateEigenGrid()">
                        <option value="2">2 x 2</option>
                        <option value="3" selected>3 x 3</option>
                        <option value="4">4 x 4</option>
                    </select>
                </div>
                <div>
                    <h3>Input Matrix A</h3>
                    <div id="eigen-matrix-container" style="margin-top:1rem;"></div>
                    <div style="display:flex; gap:0.75rem;">
                        <button class="solve-btn" onclick="executeEigen()">Compute Eigenvalues</button>
                        <button class="solve-btn" onclick="executeSVD()" style="background-color:#9B5DE5; box-shadow:0 4px 12px rgba(155, 93, 229, 0.3);">Compute SVD</button>
                    </div>
                    <div class="error-message" id="eigen-error-box"></div>
                </div>
                <div class="controls">
                    <h3>Results</h3>
                    <div class="equation-block" id="eigen-result-box"></div>
                </div>
            </div>
        </div>

        <!-- PCA & Regression Tab -->
        <div id="tab-pca-reg" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Interactive PCA & Linear Regression</span>
                    <div>
                        <button class="tab-btn active" id="btn-fit-reg" onclick="togglePCAOption('reg')" style="padding:0.4rem 0.8rem; font-size:0.85rem;">Linear Regression</button>
                        <button class="tab-btn active" id="btn-fit-pca" onclick="togglePCAOption('pca')" style="padding:0.4rem 0.8rem; font-size:0.85rem; margin-left:0.3rem;">PCA Axes</button>
                        <button class="solve-btn" onclick="clearPCAPoints()" style="padding:0.4rem 0.8rem; font-size:0.85rem; margin-left:1rem; margin-top:0; background-color:var(--danger); box-shadow:none;">Clear Points</button>
                    </div>
                </div>
                <div class="visual-area" style="min-height:450px;">
                    <canvas id="pca-canvas" class="pca-canvas-elem" width="520" height="420" onclick="handlePCACanvasClick(event)"></canvas>
                </div>
                <div class="controls">
                    <div>
                        <h3>Interactive Plotting Lab</h3>
                        <p style="color:var(--text-muted); font-size:0.9rem; margin-top:0.5rem; margin-bottom:1.5rem;">Click on the grid to add data points. The coordinates are processed in real-time by the KairoMath statistics module.</p>
                    </div>
                    <div class="pca-info">
                        <div style="font-weight:600; color:var(--accent);">Active Dataset Status</div>
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.9rem;">Points Plotted: <span id="pca-point-count" style="font-weight:600; color:#FFF;">0</span></div>
                        <div id="pca-math-details" style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; color:var(--text-muted); border-top:1px solid var(--border-color); padding-top:0.5rem; margin-top:0.5rem;">
                            Fit at least 2 points to start calculations.
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 3D Transform Tab -->
        <div id="tab-transform3d" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <span class="card-title">Interactive 3D Transform Workspace</span>
                    <span class="badge">TRS Composition Matrix</span>
                </div>
                <div class="visual-area">
                    <div class="canvas-container">
                        <canvas id="cube-canvas" width="400" height="400"></canvas>
                    </div>
                </div>
                <div class="controls">
                    <div>
                        <div class="control-group">
                            <div class="control-label"><span>Translation X</span><span class="control-value" id="val-tx">0.0</span></div>
                            <input type="range" id="slider-tx" min="-50" max="50" value="0" oninput="updateCube()">
                        </div>
                        <div class="control-group">
                            <div class="control-label"><span>Rotation Y (Yaw)</span><span class="control-value" id="val-ry">35°</span></div>
                            <input type="range" id="slider-ry" min="0" max="360" value="35" oninput="updateCube()">
                        </div>
                        <div class="control-group">
                            <div class="control-label"><span>Rotation X (Pitch)</span><span class="control-value" id="val-rx">30°</span></div>
                            <input type="range" id="slider-rx" min="0" max="360" value="30" oninput="updateCube()">
                        </div>
                        <div class="control-group">
                            <div class="control-label"><span>Scale</span><span class="control-value" id="val-scale">1.0</span></div>
                            <input type="range" id="slider-scale" min="5" max="25" value="10" oninput="updateCube()">
                        </div>
                    </div>
                    <div class="equation-block">
                        <div class="eq-line">TRS Matrix (computed via C++ Transform):</div>
                        <div class="eq-line" style="font-size:0.75rem; white-space:pre; line-height:1.3;" id="matrix-display">Calculating...</div>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        const API_BASE = "http://localhost:8080/api";
        let isServerConnected = false;

        async function testAPIConnection() {
            try {
                const response = await fetch(`${API_BASE}/project`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ax: 1, ay: 0, bx: 1, by: 0 })
                });
                if (response.ok) {
                    isServerConnected = true;
                    document.getElementById('status-dot').className = "status-dot online";
                    document.getElementById('status-text').innerText = "C++ Engine: Connected (Live C++ Math Library)";
                    updateVectorLab();
                }
            } catch(e) {}
        }
        testAPIConnection();

        async function fetchAPI(endpoint, payload, fallbackFunc) {
            if (isServerConnected) {
                try {
                    const response = await fetch(`${API_BASE}/${endpoint}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    if (response.ok) {
                        const data = await response.json();
                        if (data.status === "success") return data;
                    }
                } catch(e) {}
            }
            return fallbackFunc(payload);
        }

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            const btnIndex = {
                'tab-vectors': 0, 'tab-solve': 1, 'tab-decompose': 2,
                'tab-eigen-svd': 3, 'tab-pca-reg': 4, 'tab-transform3d': 5
            }[tabId];
            document.querySelectorAll('.tab-btn')[btnIndex].classList.add('active');

            if (tabId === 'tab-vectors') updateVectorLab();
            else if (tabId === 'tab-solve') recreateSolveGrid();
            else if (tabId === 'tab-decompose') recreateDecompGrid();
            else if (tabId === 'tab-eigen-svd') recreateEigenGrid();
            else if (tabId === 'tab-pca-reg') drawPCACanvas();
            else if (tabId === 'tab-transform3d') updateCube();
        }

        // Vector Lab Logic
        let vectorLabMode = 'projection';

        function setVectorLabMode(mode) {
            vectorLabMode = mode;
            document.getElementById('btn-mode-proj').classList.toggle('active', mode === 'projection');
            document.getElementById('btn-mode-refr').classList.toggle('active', mode === 'refraction');
            document.getElementById('ctrl-projection').style.display = mode === 'projection' ? 'block' : 'none';
            document.getElementById('ctrl-refraction').style.display = mode === 'refraction' ? 'block' : 'none';
            
            const isProj = mode === 'projection';
            document.getElementById('vec-axis').style.display = isProj ? 'block' : 'none';
            document.getElementById('vec-b').style.display = isProj ? 'block' : 'none';
            document.getElementById('vec-a').style.display = isProj ? 'block' : 'none';
            document.getElementById('vec-proj').style.display = isProj ? 'block' : 'none';
            document.getElementById('vec-helper').style.display = isProj ? 'block' : 'none';
            document.getElementById('normal-line').style.display = isProj ? 'none' : 'block';
            document.getElementById('refract-boundary').style.display = isProj ? 'none' : 'block';
            document.getElementById('medium2-rect').style.display = isProj ? 'none' : 'block';
            document.getElementById('ray-inc').style.display = isProj ? 'none' : 'block';
            document.getElementById('ray-refl').style.display = isProj ? 'none' : 'block';
            document.getElementById('ray-refr').style.display = isProj ? 'none' : 'block';
            updateVectorLab();
        }

        async function updateVectorLab() {
            if (vectorLabMode === 'projection') {
                const aAng = parseFloat(document.getElementById('slider-vec-a-ang').value) * Math.PI / 180;
                const aMag = parseFloat(document.getElementById('slider-vec-a-mag').value);
                const bAng = parseFloat(document.getElementById('slider-vec-b-ang').value) * Math.PI / 180;
                const bMag = parseFloat(document.getElementById('slider-vec-b-mag').value);

                document.getElementById('val-vec-a-ang').innerText = document.getElementById('slider-vec-a-ang').value + '°';
                document.getElementById('val-vec-a-mag').innerText = document.getElementById('slider-vec-a-mag').value;
                document.getElementById('val-vec-b-ang').innerText = document.getElementById('slider-vec-b-ang').value + '°';
                document.getElementById('val-vec-b-mag').innerText = document.getElementById('slider-vec-b-mag').value;

                const ax = aMag * Math.cos(aAng);
                const ay = -aMag * Math.sin(aAng);
                const bx = bMag * Math.cos(bAng);
                const by = -bMag * Math.sin(bAng);

                document.getElementById('vec-a').setAttribute('x2', ax);
                document.getElementById('vec-a').setAttribute('y2', ay);
                document.getElementById('vec-b').setAttribute('x2', bx);
                document.getElementById('vec-b').setAttribute('y2', by);

                if (bMag > 0) {
                    const nx = bx / bMag; const ny = by / bMag;
                    document.getElementById('vec-axis').setAttribute('x1', -nx * 225);
                    document.getElementById('vec-axis').setAttribute('y1', -ny * 225);
                    document.getElementById('vec-axis').setAttribute('x2', nx * 225);
                    document.getElementById('vec-axis').setAttribute('y2', ny * 225);
                } else {
                    document.getElementById('vec-axis').setAttribute('x1', 0);
                    document.getElementById('vec-axis').setAttribute('y1', 0);
                    document.getElementById('vec-axis').setAttribute('x2', 0);
                    document.getElementById('vec-axis').setAttribute('y2', 0);
                }

                const res = await fetchAPI('project', { ax, ay: -ay, bx, by: -by }, (p) => {
                    const dot = p.ax * p.bx + p.ay * p.by;
                    const b_len_sq = p.bx * p.bx + p.by * p.by;
                    const factor = b_len_sq > 1e-6 ? dot / b_len_sq : 0;
                    return { status: "success", px: p.bx * factor, py: p.by * factor, dot, b_len_sq };
                });

                const px = res.px; const py = -res.py;
                document.getElementById('vec-proj').setAttribute('x2', px);
                document.getElementById('vec-proj').setAttribute('y2', py);
                document.getElementById('vec-helper').setAttribute('x1', ax);
                document.getElementById('vec-helper').setAttribute('y1', ay);
                document.getElementById('vec-helper').setAttribute('x2', px);
                document.getElementById('vec-helper').setAttribute('y2', py);

                document.getElementById('vector-equation-block').innerHTML = `
                    <div class="eq-line"><span class="dot-a">A (Target)</span> = (${(ax/100).toFixed(2)}, ${(-ay/100).toFixed(2)})</div>
                    <div class="eq-line"><span class="dot-b">B (Axis)</span> = (${(bx/100).toFixed(2)}, ${(-by/100).toFixed(2)})</div>
                    <div class="eq-line">Formula: Project(A, B) = B * (A.B / B.B)</div>
                    <div class="eq-line">A.B = ${(res.dot/10000).toFixed(3)}, B.B = ${(res.b_len_sq/10000).toFixed(3)}</div>
                    <div class="eq-line"><span class="dot-proj">Projected Vector</span> = (${(px/100).toFixed(2)}, ${(-py/100).toFixed(2)})</div>
                `;
            } else {
                const rayAngDeg = parseFloat(document.getElementById('slider-ray-ang').value);
                const rayAng = rayAngDeg * Math.PI / 180;
                const n1 = parseFloat(document.getElementById('slider-ray-n1').value) / 10;
                const n2 = parseFloat(document.getElementById('slider-ray-n2').value) / 10;

                document.getElementById('val-ray-ang').innerText = rayAngDeg + '°';
                document.getElementById('val-ray-n1').innerText = n1.toFixed(1);
                document.getElementById('val-ray-n2').innerText = n2.toFixed(1);

                const ix = Math.sin(rayAng); const iy = Math.cos(rayAng);
                const rayLen = 160;
                document.getElementById('ray-inc').setAttribute('x1', -ix * rayLen);
                document.getElementById('ray-inc').setAttribute('y1', -iy * rayLen);
                document.getElementById('ray-inc').setAttribute('x2', 0);
                document.getElementById('ray-inc').setAttribute('y2', 0);

                const eta = n1 / n2;

                const res = await fetchAPI('refract', { ix, iy: -iy, iz: 0, nx: 0, ny: 1, nz: 0, eta }, (p) => {
                    const sinTheta2 = p.eta * p.ix;
                    if (Math.abs(sinTheta2) > 1.0) return { status: "success", rx: p.ix, ry: p.iy, tir: true };
                    const cosTheta2 = Math.sqrt(1.0 - sinTheta2 * sinTheta2);
                    return { status: "success", rx: p.ix, ry: p.iy, refr_x: sinTheta2, refr_y: -cosTheta2, tir: false };
                });

                const rx = res.rx; const ry = -res.ry;
                document.getElementById('ray-refl').setAttribute('x2', rx * rayLen);
                document.getElementById('ray-refl').setAttribute('y2', ry * rayLen);

                if (res.tir) {
                    document.getElementById('tir-alert').style.display = 'block';
                    document.getElementById('ray-refr').style.display = 'none';
                } else {
                    document.getElementById('tir-alert').style.display = 'none';
                    document.getElementById('ray-refr').style.display = 'block';
                    document.getElementById('ray-refr').setAttribute('x2', res.refr_x * rayLen);
                    document.getElementById('ray-refr').setAttribute('y2', -res.refr_y * rayLen);
                }

                const critAngle = n2 < n1 ? Math.asin(n2 / n1) * 180 / Math.PI : null;
                document.getElementById('vector-equation-block').innerHTML = `
                    <div class="eq-line">η = n₁ / n₂ = ${eta.toFixed(3)}</div>
                    <div class="eq-line"><span class="dot-a">Incident Vector</span> = (${ix.toFixed(2)}, ${(-iy).toFixed(2)})</div>
                    <div class="eq-line"><span class="dot-reflect">Reflected Vector</span> = (${rx.toFixed(2)}, ${(-ry).toFixed(2)})</div>
                    <div class="eq-line"><span class="dot-refract">Refracted Vector</span> = ${res.tir ? 'TIR' : `(${res.refr_x.toFixed(2)}, ${res.refr_y.toFixed(2)})`}</div>
                    <div class="eq-line">Critical Angle: ${critAngle ? critAngle.toFixed(1) + '°' : 'None'}</div>
                `;
            }
        }

        // Linear Solver Lab Logic
        function recreateSolveGrid() {
            const dim = parseInt(document.getElementById('solve-dim').value);
            const matContainer = document.getElementById('solve-matrix-container');
            let html = '<div class="matrix-grid" style="grid-template-columns: repeat(' + dim + ', 1fr);">';
            for (let r = 0; r < dim; ++r) {
                for (let c = 0; c < dim; ++c) html += `<input type="text" id="solve-A-${r}-${c}" class="matrix-cell" value="0">`;
            }
            matContainer.innerHTML = html + '</div>';

            const bContainer = document.getElementById('solve-b-container');
            let bHtml = '<div class="matrix-grid" style="grid-template-columns: 1fr;">';
            for (let r = 0; r < dim; ++r) bHtml += `<input type="text" id="solve-b-${r}" class="matrix-cell" value="0">`;
            bContainer.innerHTML = bHtml + '</div>';

            applySolvePreset();
        }

        function applySolvePreset() {
            const dim = parseInt(document.getElementById('solve-dim').value);
            const preset = document.getElementById('solve-preset').value;
            for (let r = 0; r < dim; ++r) {
                for (let c = 0; c < dim; ++c) {
                    let val = (r === c) ? (r + 2) : 1;
                    if (preset === 'singular' && r === dim - 1 && dim > 1) val = (r+1) + (c+1);
                    if (preset === 'hilbert') val = 1.0 / (r + c + 1.0);
                    document.getElementById(`solve-A-${r}-${c}`).value = val.toFixed(2);
                }
                document.getElementById(`solve-b-${r}`).value = (r + 2).toFixed(2);
            }
            executeLinearSolve();
        }

        async function executeLinearSolve() {
            const dim = parseInt(document.getElementById('solve-dim').value);
            const matrix = []; const b = [];
            for (let r = 0; r < dim; ++r) {
                matrix[r] = [];
                for (let c = 0; c < dim; ++c) matrix[r][c] = parseFloat(document.getElementById(`solve-A-${r}-${c}`).value) || 0;
                b[r] = parseFloat(document.getElementById(`solve-b-${r}`).value) || 0;
            }

            const errBox = document.getElementById('solve-error-box'); errBox.style.display = 'none';
            const res = await fetchAPI('solve', { matrix, b }, (p) => {
                const n = p.matrix.length;
                if (n === 2) {
                    const det = p.matrix[0][0]*p.matrix[1][1] - p.matrix[0][1]*p.matrix[1][0];
                    if (Math.abs(det) < 1e-9) return { status: "success", solve_error: "Singular matrix", ref: p.matrix, rref: p.matrix };
                    const x = [
                        (p.b[0]*p.matrix[1][1] - p.matrix[0][1]*p.b[1]) / det,
                        (p.matrix[0][0]*p.b[1] - p.b[0]*p.matrix[1][0]) / det
                    ];
                    return { status: "success", x, ref: p.matrix, rref: [[1, 0], [0, 1]] };
                }
                return { status: "success", solve_error: "Local JS fallback only supports 2x2 systems. Please start Python API server to run C++ math engine.", ref: p.matrix, rref: p.matrix };
            });

            const solveX = document.getElementById('solve-x-result');
            if (res.solve_error) {
                errBox.style.display = 'block'; errBox.innerText = "Error: " + res.solve_error;
                solveX.innerHTML = `<span style="color:var(--danger)">No Unique Solution</span>`;
            } else {
                solveX.innerHTML = res.x.map((val, idx) => `<div class="eq-line">x[${idx}] = <span style="color:#FFF; font-weight:bold;">${val.toFixed(4)}</span></div>`).join('');
            }
            renderOutputMatrix('solve-ref-matrix', res.ref);
            renderOutputMatrix('solve-rref-matrix', res.rref);
        }

        function renderOutputMatrix(containerId, matrix) {
            const container = document.getElementById(containerId);
            const cols = matrix[0] ? matrix[0].length : 0;
            let html = `<div class="output-matrix" style="grid-template-columns: repeat(${cols}, 1fr);">`;
            for (let r = 0; r < matrix.length; ++r) {
                for (let c = 0; c < cols; ++c) html += `<div class="output-matrix-cell">${matrix[r][c].toFixed(3)}</div>`;
            }
            container.innerHTML = html + `</div>`;
        }

        // Matrix Decomposition Lab Logic
        function recreateDecompGrid() {
            const dim = parseInt(document.getElementById('decomp-dim').value);
            const container = document.getElementById('decomp-matrix-container');
            let html = '<div class="matrix-grid" style="grid-template-columns: repeat(' + dim + ', 1fr);">';
            for (let r = 0; r < dim; ++r) {
                for (let c = 0; c < dim; ++c) {
                    let val = (r === c) ? (r + 4.0) : 1.0;
                    html += `<input type="text" id="decomp-A-${r}-${c}" class="matrix-cell" value="${val.toFixed(2)}">`;
                }
            }
            container.innerHTML = html + '</div>';
            executeDecomposition();
        }

        async function executeDecomposition() {
            const dim = parseInt(document.getElementById('decomp-dim').value);
            const method = document.getElementById('decomp-method').value;
            const matrix = [];
            for (let r = 0; r < dim; ++r) {
                matrix[r] = [];
                for (let c = 0; c < dim; ++c) matrix[r][c] = parseFloat(document.getElementById(`decomp-A-${r}-${c}`).value) || 0;
            }

            const errBox = document.getElementById('decomp-error-box'); errBox.style.display = 'none';
            const resultBox = document.getElementById('decomp-result-box');

            const res = await fetchAPI('decompose', { method, matrix }, (p) => {
                return { status: "error", message: "Live C++ backend required for decompositions." };
            });

            if (res.status === "error" || res.message) {
                errBox.style.display = 'block'; errBox.innerText = res.message || "Failed";
                resultBox.innerHTML = "<div style='color:var(--text-muted);'>C++ backend required. Please run server.py.</div>";
                return;
            }

            let html = "";
            if (method === 'lu') {
                html = `<div>L Matrix:</div>${matrixToHtml(res.L)}<div>U Matrix:</div>${matrixToHtml(res.U)}`;
            } else if (method === 'lup') {
                html = `<div>L Matrix:</div>${matrixToHtml(res.L)}<div>U Matrix:</div>${matrixToHtml(res.U)}<div>P Vector: [${res.P.join(', ')}]</div>`;
            } else if (method === 'qr') {
                html = `<div>Q Matrix:</div>${matrixToHtml(res.Q)}<div>R Matrix:</div>${matrixToHtml(res.R)}`;
            } else if (method === 'cholesky') {
                html = `<div>L Matrix:</div>${matrixToHtml(res.L)}`;
            } else if (method === 'ldlt') {
                html = `<div>L Matrix:</div>${matrixToHtml(res.L)}<div>D Vector: [${res.D.map(v => v.toFixed(3)).join(', ')}]</div>`;
            }
            resultBox.innerHTML = html;
        }

        function matrixToHtml(matrix) {
            const cols = matrix[0] ? matrix[0].length : 0;
            let html = `<div class="output-matrix" style="grid-template-columns: repeat(${cols}, 1fr); margin-bottom:1rem;">`;
            for (let r = 0; r < matrix.length; ++r) {
                for (let c = 0; c < cols; ++c) html += `<div class="output-matrix-cell">${matrix[r][c].toFixed(3)}</div>`;
            }
            return html + `</div>`;
        }

        // Eigenvalues & SVD Lab Logic
        function recreateEigenGrid() {
            const dim = parseInt(document.getElementById('eigen-dim').value);
            const container = document.getElementById('eigen-matrix-container');
            let html = '<div class="matrix-grid" style="grid-template-columns: repeat(' + dim + ', 1fr);">';
            for (let r = 0; r < dim; ++r) {
                for (let c = 0; c < dim; ++c) {
                    let val = (r === c) ? (r + 3.0) : 1.5;
                    html += `<input type="text" id="eigen-A-${r}-${c}" class="matrix-cell" value="${val.toFixed(2)}">`;
                }
            }
            container.innerHTML = html + '</div>';
            executeEigen();
        }

        async function executeEigen() {
            const dim = parseInt(document.getElementById('eigen-dim').value);
            const matrix = [];
            for (let r = 0; r < dim; ++r) {
                matrix[r] = [];
                for (let c = 0; c < dim; ++c) matrix[r][c] = parseFloat(document.getElementById(`eigen-A-${r}-${c}`).value) || 0;
            }

            const errBox = document.getElementById('eigen-error-box'); errBox.style.display = 'none';
            const resultBox = document.getElementById('eigen-result-box');

            const res = await fetchAPI('eigen', { matrix }, (p) => {
                return { status: "error", message: "Live C++ backend required for Eigen values." };
            });

            if (res.status === "error" || res.message) {
                errBox.style.display = 'block'; errBox.innerText = res.message || "Failed";
                resultBox.innerHTML = "<div style='color:var(--text-muted);'>C++ backend required. Please run server.py.</div>";
                return;
            }

            resultBox.innerHTML = `
                <div>Eigenvalues: [ ${res.eigenvalues.map(v => v.toFixed(4)).join(', ')} ]</div>
                <div style="margin-top:1rem;">Eigenvectors V (columns):</div>
                ${matrixToHtml(res.eigenvectors)}
            `;
        }

        async function executeSVD() {
            const dim = parseInt(document.getElementById('eigen-dim').value);
            const matrix = [];
            for (let r = 0; r < dim; ++r) {
                matrix[r] = [];
                for (let c = 0; c < dim; ++c) matrix[r][c] = parseFloat(document.getElementById(`eigen-A-${r}-${c}`).value) || 0;
            }

            const errBox = document.getElementById('eigen-error-box'); errBox.style.display = 'none';
            const resultBox = document.getElementById('eigen-result-box');

            const res = await fetchAPI('svd', { matrix }, (p) => {
                return { status: "error", message: "Live C++ backend required for SVD." };
            });

            if (res.status === "error" || res.message) {
                errBox.style.display = 'block'; errBox.innerText = res.message || "Failed";
                resultBox.innerHTML = "<div style='color:var(--text-muted);'>C++ backend required. Please run server.py.</div>";
                return;
            }

            resultBox.innerHTML = `
                <div>Left Singular Vectors U:</div>${matrixToHtml(res.U)}
                <div>Singular Values: [ ${res.Sigma.map(v => v.toFixed(4)).join(', ')} ]</div>
                <div style="margin-top:1rem;">Right Singular Vectors V:</div>${matrixToHtml(res.V)}
            `;
        }

        // PCA & Regression Canvas Lab Logic
        const pcaPoints = [];
        let pcaOptions = { reg: true, pca: true };

        function togglePCAOption(opt) {
            pcaOptions[opt] = !pcaOptions[opt];
            document.getElementById(`btn-fit-${opt}`).classList.toggle('active', pcaOptions[opt]);
            drawPCACanvas();
        }

        function clearPCAPoints() {
            pcaPoints.length = 0;
            document.getElementById('pca-point-count').innerText = 0;
            document.getElementById('pca-math-details').innerText = "Fit at least 2 points to start calculations.";
            drawPCACanvas();
        }

        function handlePCACanvasClick(e) {
            const canvas = document.getElementById('pca-canvas');
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;

            const scale = 40;
            const xVal = (clickX - canvas.width/2) / scale;
            const yVal = -(clickY - canvas.height/2) / scale;

            pcaPoints.push({ x: xVal, y: yVal });
            document.getElementById('pca-point-count').innerText = pcaPoints.length;
            drawPCACanvas();
        }

        async function drawPCACanvas() {
            const canvas = document.getElementById('pca-canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const scale = 40;
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            ctx.strokeStyle = '#1E2433';
            ctx.lineWidth = 1;
            for (let x = -6; x <= 6; ++x) {
                ctx.beginPath(); ctx.moveTo(cx + x*scale, 0); ctx.lineTo(cx + x*scale, canvas.height); ctx.stroke();
            }
            for (let y = -5; y <= 5; ++y) {
                ctx.beginPath(); ctx.moveTo(0, cy + y*scale); ctx.lineTo(canvas.width, cy + y*scale); ctx.stroke();
            }

            ctx.strokeStyle = '#2D3748';
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height); ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();

            ctx.fillStyle = '#FFF';
            for (let p of pcaPoints) {
                ctx.beginPath(); ctx.arc(cx + p.x*scale, cy - p.y*scale, 5, 0, 2*Math.PI); ctx.fill();
            }

            if (pcaPoints.length < 2) return;

            let regRes = null;
            let pcaRes = null;

            if (pcaOptions.reg) {
                const matrix = pcaPoints.map(p => [p.x, 1.0]);
                const y = pcaPoints.map(p => p.y);
                regRes = await fetchAPI('regression', { matrix, y }, (p) => {
                    const n = p.matrix.length;
                    let sumX=0, sumY=0, sumXX=0, sumXY=0;
                    for(let i=0; i<n; ++i) {
                        sumX += p.matrix[i][0]; sumY += p.y[i];
                        sumXX += p.matrix[i][0]*p.matrix[i][0]; sumXY += p.matrix[i][0]*p.y[i];
                    }
                    const denom = (n*sumXX - sumX*sumX);
                    const slope = denom !== 0 ? (n*sumXY - sumX*sumY)/denom : 0;
                    const intercept = (sumY - slope*sumX)/n;
                    return { status: "success", beta: [slope, intercept] };
                });
            }

            if (pcaOptions.pca) {
                const matrix = pcaPoints.map(p => [p.x, p.y]);
                pcaRes = await fetchAPI('pca', { matrix }, (p) => {
                    const n = p.matrix.length;
                    let mX = 0, mY = 0;
                    for (let row of p.matrix) { mX += row[0]; mY += row[1]; }
                    mX /= n; mY /= n;
                    let covXX = 0, covYY = 0, covXY = 0;
                    for (let row of p.matrix) {
                        const dx = row[0] - mX;
                        const dy = row[1] - mY;
                        covXX += dx * dx; covYY += dy * dy; covXY += dx * dy;
                    }
                    covXX /= (n - 1); covYY /= (n - 1); covXY /= (n - 1);
                    const tr = covXX + covYY;
                    const det = covXX * covYY - covXY * covXY;
                    const desc = Math.sqrt(Math.max(0, tr*tr/4 - det));
                    const l1 = tr/2 + desc;
                    const l2 = Math.max(0, tr/2 - desc);
                    
                    const v1 = [covXY, l1 - covXX];
                    const len = Math.sqrt(v1[0]*v1[0] + v1[1]*v1[1]);
                    if (len > 1e-5) { v1[0]/=len; v1[1]/=len; } else { v1[0]=1; v1[1]=0; }
                    return { status: "success", explained_variance: [l1, l2], components: [[v1[0], -v1[1]], [v1[1], v1[0]]] };
                });
            }

            if (regRes) {
                const slope = regRes.beta[0]; const intercept = regRes.beta[1];
                ctx.strokeStyle = 'var(--reflect)'; ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.moveTo(cx - 6*scale, cy - (slope*-6+intercept)*scale);
                ctx.lineTo(cx + 6*scale, cy - (slope*6+intercept)*scale);
                ctx.stroke();
            }

            if (pcaRes && pcaRes.status !== "error") {
                let mX=0, mY=0;
                for(let p of pcaPoints) { mX+=p.x; mY+=p.y; }
                mX/=pcaPoints.length; mY/=pcaPoints.length;

                ctx.fillStyle = 'var(--project)';
                ctx.beginPath(); ctx.arc(cx + mX*scale, cy - mY*scale, 6, 0, 2*Math.PI); ctx.fill();

                const len1 = Math.sqrt(pcaRes.explained_variance[0]) * scale * 1.5;
                ctx.strokeStyle = 'var(--vector-b)'; ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(cx + mX*scale, cy - mY*scale);
                ctx.lineTo(cx + mX*scale + pcaRes.components[0][0]*len1, cy - mY*scale - pcaRes.components[1][0]*len1);
                ctx.stroke();

                const len2 = Math.sqrt(pcaRes.explained_variance[1]) * scale * 1.5;
                ctx.strokeStyle = 'var(--project)'; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.moveTo(cx + mX*scale, cy - mY*scale);
                ctx.lineTo(cx + mX*scale + pcaRes.components[0][1]*len2, cy - mY*scale - pcaRes.components[1][1]*len2);
                ctx.stroke();
            }

            let infoHtml = "";
            if (regRes) infoHtml += `<div>Regression: y = ${regRes.beta[0].toFixed(3)}x + ${regRes.beta[1].toFixed(3)}</div>`;
            if (pcaRes && pcaRes.status !== "error") {
                infoHtml += `<div>PC1 Variance: ${pcaRes.explained_variance[0].toFixed(3)}</div>`;
                infoHtml += `<div>PC2 Variance: ${pcaRes.explained_variance[1].toFixed(3)}</div>`;
            }
            document.getElementById('pca-math-details').innerHTML = infoHtml;
        }

        // 3D Transform Cube Visualizer Logic
        const canvas = document.getElementById('cube-canvas');
        const ctx = canvas.getContext('2d');
        const sliderTx = document.getElementById('slider-tx');
        const sliderRy = document.getElementById('slider-ry');
        const sliderRx = document.getElementById('slider-rx');
        const sliderScale = document.getElementById('slider-scale');

        const vertices = [
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ];
        const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];

        function multiplyMatrixVector(m, v) {
            const out = [0, 0, 0, 0];
            for (let i = 0; i < 4; i++) {
                out[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2] + m[i][3]*v[3];
            }
            return out;
        }

        async function updateCube() {
            const tx = parseFloat(sliderTx.value);
            const ry = parseFloat(sliderRy.value) * Math.PI / 180;
            const rx = parseFloat(sliderRx.value) * Math.PI / 180;
            const scale = parseFloat(sliderScale.value) / 10;

            document.getElementById('val-tx').innerText = (tx/10).toFixed(1);
            document.getElementById('val-ry').innerText = sliderRy.value + '°';
            document.getElementById('val-rx').innerText = sliderRx.value + '°';
            document.getElementById('val-scale').innerText = scale.toFixed(1);

            const res = await fetchAPI('transform', {
                tx: tx/20, ty: 0.0, tz: 0.0, rx, ry, rz: 0.0, sx: scale, sy: scale, sz: scale
            }, (p) => {
                const cosY = Math.cos(p.ry); const sinY = Math.sin(p.ry);
                const cosX = Math.cos(p.rx); const sinX = Math.sin(p.rx);
                return {
                    status: "success",
                    matrix: [
                        [cosY*p.sx, sinY*sinX*p.sy, sinY*cosX*p.sz, p.tx],
                        [0, cosX*p.sy, -sinX*p.sz, p.ty],
                        [-sinY*p.sx, cosY*sinX*p.sy, cosY*cosX*p.sz, p.tz],
                        [0, 0, 0, 1]
                    ]
                };
            });

            const trsMat = res.matrix;
            let mStr = "";
            for (let r = 0; r < 4; r++) {
                mStr += "[ ";
                for (let c = 0; c < 4; c++) mStr += (trsMat[r][c] >= 0 ? " " : "") + trsMat[r][c].toFixed(2) + "  ";
                mStr = mStr.trim() + " ]\n";
            }
            document.getElementById('matrix-display').innerText = mStr;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#1E2433'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(canvas.width/2, 0); ctx.lineTo(canvas.width/2, canvas.height); ctx.moveTo(0, canvas.height/2); ctx.lineTo(canvas.width, canvas.height/2); ctx.stroke();

            const projected = [];
            const distance = 4; const scaleProj = 200;
            for (let i = 0; i < vertices.length; i++) {
                const v = [...vertices[i], 1.0];
                const transformed = multiplyMatrixVector(trsMat, v);
                const z = transformed[2] + distance;
                let x2d = (transformed[0] / z) * scaleProj + canvas.width / 2;
                let y2d = (transformed[1] / z) * scaleProj + canvas.height / 2;
                projected.push([x2d, y2d]);
            }

            ctx.strokeStyle = '#4ECDC4'; ctx.lineWidth = 2.5;
            for (let i = 0; i < edges.length; i++) {
                const p0 = projected[edges[i][0]]; const p1 = projected[edges[i][1]];
                ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]); ctx.stroke();
            }
        }
    </script>
</body>
</html>
)HTML";

    file.close();
    std::cout << "Successfully generated " << filepath << "\\n";
}"""

# Replace in content
updated_content = content[:start_idx] + new_html_generator + content[end_idx:]

with open(cpp_file_path, "w") as f:
    f.write(updated_content)

print("Update completed successfully.")
