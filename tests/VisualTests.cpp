#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>

import Kairo.Foundation.Math.Vector;
import Kairo.Foundation.Math.Matrix;
import Kairo.Foundation.Math.Quaternion;
import Kairo.Foundation.Math.Transform;

using namespace kairo::foundation::math;

// ASCII Renderer helper class for rendering 3D wireframe cube to console
class AsciiRenderer
{
public:
    int width;
    int height;
    std::vector<std::string> buffer;

    AsciiRenderer(int w, int h) : width(w), height(h)
    {
        Clear();
    }

    void Clear()
    {
        buffer.assign(height, std::string(width, ' '));
    }

    void DrawPixel(int x, int y, char c)
    {
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            buffer[y][x] = c;
        }
    }

    void DrawLine(int x0, int y0, int x1, int y1, char c)
    {
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            DrawPixel(x0, y0, c);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    void Render()
    {
        // Cursor to top-left
        std::cout << "\033[H";
        for (const auto& line : buffer)
        {
            std::cout << line << "\n";
        }
    }
};

// Writes the interactive HTML visualizer page
void GenerateHTMLVisualizer(const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Failed to generate visual_tests.html\n";
        return;
    }

    file << R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KairoMath Visual Tests</title>
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
            margin: 0 auto 3rem auto;
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

        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 3rem;
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
            min-height: 400px;
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
    </style>
</head>
<body>

    <header>
        <h1>KairoMath Visualizer</h1>
        <div class="subtitle">Interactive verification of C++ module math operations</div>
    </header>

    <div class="container">
        
        <!-- CARD 1: Vector Projection -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Vector Projection</span>
                <span class="badge">Non-unit & Zero-length Axes</span>
            </div>
            
            <div class="visual-area">
                <svg id="projection-svg" width="400" height="400" viewBox="-200 -200 400 400">
                    <!-- Grid Lines -->
                    <line x1="-200" y1="0" x2="200" y2="0" stroke="#1E2433" stroke-width="1" />
                    <line x1="0" y1="-200" x2="0" y2="200" stroke="#1E2433" stroke-width="1" />
                    <!-- Circle markers -->
                    <circle cx="0" cy="0" r="4" fill="#FFF" />
                    
                    <!-- Axis projection line -->
                    <line id="axis-line" x1="-200" y1="0" x2="200" y2="0" stroke="#2D3748" stroke-dasharray="4,4" stroke-width="1.5" />
                    
                    <!-- Vectors -->
                    <line id="arrow-b" x1="0" y1="0" x2="120" y2="40" stroke="var(--vector-b)" stroke-width="3.5" marker-end="url(#arrow-head-b)" />
                    <line id="arrow-a" x1="0" y1="0" x2="60" y2="100" stroke="var(--vector-a)" stroke-width="3.5" marker-end="url(#arrow-head-a)" />
                    <line id="arrow-proj" x1="0" y1="0" x2="0" y2="0" stroke="var(--project)" stroke-width="4" marker-end="url(#arrow-head-proj)" />
                    
                    <!-- Helper dotted lines from A to projection -->
                    <line id="projection-helper" x1="0" y1="0" x2="0" y2="0" stroke="#718096" stroke-dasharray="3,3" stroke-width="1.5" />
                    
                    <!-- Markers Definition -->
                    <defs>
                        <marker id="arrow-head-a" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--vector-a)" />
                        </marker>
                        <marker id="arrow-head-b" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--vector-b)" />
                        </marker>
                        <marker id="arrow-head-proj" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                            <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--project)" />
                        </marker>
                    </defs>
                </svg>
            </div>
            
            <div class="controls">
                <div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Vector A (Target) Angle</span>
                            <span class="control-value" id="val-a-ang">60°</span>
                        </div>
                        <input type="range" id="slider-a-ang" min="0" max="360" value="60">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Vector A Magnitude</span>
                            <span class="control-value" id="val-a-mag">120</span>
                        </div>
                        <input type="range" id="slider-a-mag" min="10" max="180" value="120">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Vector B (Projection Axis) Angle</span>
                            <span class="control-value" id="val-b-ang">15°</span>
                        </div>
                        <input type="range" id="slider-b-ang" min="0" max="360" value="15">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Vector B Magnitude (Non-Unit)</span>
                            <span class="control-value" id="val-b-mag">150</span>
                        </div>
                        <input type="range" id="slider-b-mag" min="0" max="180" value="150">
                    </div>
                </div>

                <div class="equation-block">
                    <div class="eq-line"><span class="dot-a">A</span> = (<span id="eq-a-x">0</span>, <span id="eq-a-y">0</span>)</div>
                    <div class="eq-line"><span class="dot-b">B</span> = (<span id="eq-b-x">0</span>, <span id="eq-b-y">0</span>)</div>
                    <div class="eq-line">Formula: Project(A, B) = B * (A.B / B.B)</div>
                    <div class="eq-line">A.B = <span id="eq-dot">0</span>, B.B = <span id="eq-lensq">0</span></div>
                    <div class="eq-line"><span class="dot-proj">Projected</span> = (<span id="eq-proj-x">0</span>, <span id="eq-proj-y">0</span>)</div>
                </div>
            </div>
        </div>

        <!-- CARD 2: Reflection & Refraction -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Ray Reflection & Refraction</span>
                <span class="badge">TIR Guarding & Snell's Law</span>
            </div>
            
            <div class="visual-area">
                <svg id="refraction-svg" width="400" height="400" viewBox="-200 -200 400 400">
                    <!-- Medium divider (Glass/Water) -->
                    <rect x="-200" y="0" width="400" height="200" fill="#131C30" />
                    <!-- Boundary text -->
                    <text x="-190" y="-10" fill="var(--text-muted)" font-size="12">Air (n₁ = 1.0)</text>
                    <text id="txt-medium2" x="-190" y="20" fill="var(--text-muted)" font-size="12">Medium 2 (n₂ = 1.5)</text>
                    
                    <!-- Normal line -->
                    <line x1="0" y1="-180" x2="0" y2="180" stroke="var(--normal)" stroke-width="1.5" stroke-dasharray="5,5" />
                    <text x="10" y="-160" fill="var(--normal)" font-size="12">Normal</text>
                    
                    <!-- Rays -->
                    <line id="ray-incident" x1="0" y1="0" x2="0" y2="0" stroke="var(--vector-a)" stroke-width="3" />
                    <line id="ray-reflect" x1="0" y1="0" x2="0" y2="0" stroke="var(--reflect)" stroke-width="3" />
                    <line id="ray-refract" x1="0" y1="0" x2="0" y2="0" stroke="var(--refract)" stroke-width="3" />
                </svg>
            </div>
            
            <div class="controls">
                <div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Incident Ray Angle (with Normal)</span>
                            <span class="control-value" id="val-incident-ang">45°</span>
                        </div>
                        <input type="range" id="slider-incident-ang" min="0" max="89" value="45">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Medium 1 Index (n₁)</span>
                            <span class="control-value" id="val-n1">1.0</span>
                        </div>
                        <input type="range" id="slider-n1" min="10" max="30" value="10">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Medium 2 Index (n₂)</span>
                            <span class="control-value" id="val-n2">1.5</span>
                        </div>
                        <input type="range" id="slider-n2" min="10" max="30" value="15">
                    </div>
                    <div class="alert-tir" id="tir-msg">TOTAL INTERNAL REFLECTION (Refracted = 0)</div>
                </div>

                <div class="equation-block">
                    <div class="eq-line">η = n₁ / n₂ = <span id="eq-eta">0.67</span></div>
                    <div class="eq-line"><span class="dot-a">Incident</span> = (<span id="eq-inc-x">0</span>, <span id="eq-inc-y">0</span>)</div>
                    <div class="eq-line"><span class="dot-reflect">Reflected</span> = (<span id="eq-refl-x">0</span>, <span id="eq-refl-y">0</span>)</div>
                    <div class="eq-line"><span class="dot-refract">Refracted</span> = (<span id="eq-refr-x">0</span>, <span id="eq-refr-y">0</span>)</div>
                    <div class="eq-line">Critical Angle: <span id="eq-crit">41.8°</span></div>
                </div>
            </div>
        </div>

        <!-- CARD 3: 3D Transform Cube Visualizer -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">Interactive 3D Transform</span>
                <span class="badge">TRS Composition Order</span>
            </div>
            
            <div class="visual-area">
                <div class="canvas-container">
                    <canvas id="cube-canvas" width="400" height="400"></canvas>
                </div>
            </div>
            
            <div class="controls">
                <div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Translation X</span>
                            <span class="control-value" id="val-tx">0.0</span>
                        </div>
                        <input type="range" id="slider-tx" min="-50" max="50" value="0">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Rotation Y (Yaw)</span>
                            <span class="control-value" id="val-ry">35°</span>
                        </div>
                        <input type="range" id="slider-ry" min="0" max="360" value="35">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Rotation X (Pitch)</span>
                            <span class="control-value" id="val-rx">30°</span>
                        </div>
                        <input type="range" id="slider-rx" min="0" max="360" value="30">
                    </div>
                    <div class="control-group">
                        <div class="control-label">
                            <span>Scale (Uniform)</span>
                            <span class="control-value" id="val-scale">1.0</span>
                        </div>
                        <input type="range" id="slider-scale" min="5" max="25" value="10">
                    </div>
                </div>

                <div class="equation-block">
                    <div class="eq-line">Transform Matrix (M = Translation * Rotation * Scale):</div>
                    <div class="eq-line" style="font-size:0.75rem; white-space:pre;" id="matrix-display">
[ 1.00  0.00  0.00  t.x ]
[ 0.00  1.00  0.00  t.y ]
[ 0.00  0.00  1.00  t.z ]
[ 0.00  0.00  0.00  1.00 ]
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        //---------------------------------------------------------
        // Vector Projection Visualizer Logic
        //---------------------------------------------------------
        const sliderAAng = document.getElementById('slider-a-ang');
        const sliderAMag = document.getElementById('slider-a-mag');
        const sliderBAng = document.getElementById('slider-b-ang');
        const sliderBMag = document.getElementById('slider-b-mag');
        
        function updateProjection() {
            const aAng = parseFloat(sliderAAng.value) * Math.PI / 180;
            const aMag = parseFloat(sliderAMag.value);
            const bAng = parseFloat(sliderBAng.value) * Math.PI / 180;
            const bMag = parseFloat(sliderBMag.value);

            document.getElementById('val-a-ang').innerText = sliderAAng.value + '°';
            document.getElementById('val-a-mag').innerText = sliderAMag.value;
            document.getElementById('val-b-ang').innerText = sliderBAng.value + '°';
            document.getElementById('val-b-mag').innerText = sliderBMag.value;

            // Target vector A (X increases right, Y increases down in SVG - we invert Y for visual math coordinate systems)
            const ax = aMag * Math.cos(aAng);
            const ay = -aMag * Math.sin(aAng);

            // Direction vector B
            const bx = bMag * Math.cos(bAng);
            const by = -bMag * Math.sin(bAng);

            // Update SVG lines
            document.getElementById('arrow-a').setAttribute('x2', ax);
            document.getElementById('arrow-a').setAttribute('y2', ay);
            document.getElementById('arrow-b').setAttribute('x2', bx);
            document.getElementById('arrow-b').setAttribute('y2', by);

            // Draw axis guideline
            if (bMag > 0) {
                const normX = bx / bMag;
                const normY = by / bMag;
                document.getElementById('axis-line').setAttribute('x1', -normX * 200);
                document.getElementById('axis-line').setAttribute('y1', -normY * 200);
                document.getElementById('axis-line').setAttribute('x2', normX * 200);
                document.getElementById('axis-line').setAttribute('y2', normY * 200);
            } else {
                document.getElementById('axis-line').setAttribute('x1', 0);
                document.getElementById('axis-line').setAttribute('y1', 0);
                document.getElementById('axis-line').setAttribute('x2', 0);
                document.getElementById('axis-line').setAttribute('y2', 0);
            }

            // Calculation
            const dot = ax * bx + ay * by;
            const bLenSq = bx * bx + by * by;
            
            let px = 0;
            let py = 0;

            if (bLenSq > 1e-5) {
                const factor = dot / bLenSq;
                px = bx * factor;
                py = by * factor;
            }

            // Update SVG projection arrow and Helper line
            document.getElementById('arrow-proj').setAttribute('x2', px);
            document.getElementById('arrow-proj').setAttribute('y2', py);
            
            const helper = document.getElementById('projection-helper');
            helper.setAttribute('x1', ax);
            helper.setAttribute('y1', ay);
            helper.setAttribute('x2', px);
            helper.setAttribute('y2', py);

            // Display math
            document.getElementById('eq-a-x').innerText = (ax/100).toFixed(2);
            document.getElementById('eq-a-y').innerText = (-ay/100).toFixed(2);
            document.getElementById('eq-b-x').innerText = (bx/100).toFixed(2);
            document.getElementById('eq-b-y').innerText = (-by/100).toFixed(2);
            document.getElementById('eq-dot').innerText = (dot/10000).toFixed(3);
            document.getElementById('eq-lensq').innerText = (bLenSq/10000).toFixed(3);
            document.getElementById('eq-proj-x').innerText = (px/100).toFixed(2);
            document.getElementById('eq-proj-y').innerText = (-py/100).toFixed(2);
        }

        sliderAAng.oninput = updateProjection;
        sliderAMag.oninput = updateProjection;
        sliderBAng.oninput = updateProjection;
        sliderBMag.oninput = updateProjection;
        updateProjection();

        //---------------------------------------------------------
        // Ray Reflection & Refraction Visualizer Logic
        //---------------------------------------------------------
        const sliderIncAng = document.getElementById('slider-incident-ang');
        const sliderN1 = document.getElementById('slider-n1');
        const sliderN2 = document.getElementById('slider-n2');
        const tirMsg = document.getElementById('tir-msg');

        function updateRefraction() {
            const incAngDeg = parseFloat(sliderIncAng.value);
            const incAng = incAngDeg * Math.PI / 180;
            const n1 = parseFloat(sliderN1.value) / 10;
            const n2 = parseFloat(sliderN2.value) / 10;

            document.getElementById('val-incident-ang').innerText = sliderIncAng.value + '°';
            document.getElementById('val-n1').innerText = n1.toFixed(1);
            document.getElementById('val-n2').innerText = n2.toFixed(1);
            document.getElementById('txt-medium2').innerText = `Medium 2 (n₂ = ${n2.toFixed(1)})`;

            const eta = n1 / n2;
            document.getElementById('eq-eta').innerText = eta.toFixed(3);

            // Incident Vector (points towards origin)
            const ix = Math.sin(incAng);
            const iy = Math.cos(incAng); // Incident is entering from top (-y direction)
            
            // Set SVG Incident ray (draw from start to origin)
            const rayLen = 150;
            document.getElementById('ray-incident').setAttribute('x1', -ix * rayLen);
            document.getElementById('ray-incident').setAttribute('y1', -iy * rayLen);
            document.getElementById('ray-incident').setAttribute('x2', 0);
            document.getElementById('ray-incident').setAttribute('y2', 0);

            // Reflection Ray (points away from origin, y is flipped)
            const rx = ix;
            const ry = -iy;
            document.getElementById('ray-reflect').setAttribute('x1', 0);
            document.getElementById('ray-reflect').setAttribute('y1', 0);
            document.getElementById('ray-reflect').setAttribute('x2', rx * rayLen);
            document.getElementById('ray-reflect').setAttribute('y2', ry * rayLen);

            // Refraction calculations
            // Normal points up in Air: [0, 1]
            // Snell's Law: n1 * sin(theta1) = n2 * sin(theta2)
            const sinTheta2 = eta * Math.sin(incAng);
            
            const critAngle = n2 < n1 ? Math.asin(n2 / n1) * 180 / Math.PI : null;
            document.getElementById('eq-crit').innerText = critAngle ? critAngle.toFixed(1) + '°' : 'N/A';

            if (sinTheta2 > 1.0) {
                // Total Internal Reflection
                tirMsg.style.display = 'block';
                document.getElementById('ray-refract').style.display = 'none';
                document.getElementById('eq-refr-x').innerText = '0.00';
                document.getElementById('eq-refr-y').innerText = '0.00';
            } else {
                tirMsg.style.display = 'none';
                document.getElementById('ray-refract').style.display = 'block';
                
                const cosTheta2 = Math.sqrt(1.0 - sinTheta2 * sinTheta2);
                const refrX = sinTheta2;
                const refrY = cosTheta2; // Refracted ray goes downwards in medium 2 (+y)
                
                document.getElementById('ray-refract').setAttribute('x1', 0);
                document.getElementById('ray-refract').setAttribute('y1', 0);
                document.getElementById('ray-refract').setAttribute('x2', refrX * rayLen);
                document.getElementById('ray-refract').setAttribute('y2', refrY * rayLen);

                document.getElementById('eq-refr-x').innerText = refrX.toFixed(2);
                document.getElementById('eq-refr-y').innerText = (-refrY).toFixed(2);
            }

            document.getElementById('eq-inc-x').innerText = ix.toFixed(2);
            document.getElementById('eq-inc-y').innerText = (-iy).toFixed(2);
            document.getElementById('eq-refl-x').innerText = rx.toFixed(2);
            document.getElementById('eq-refl-y').innerText = (-ry).toFixed(2);
        }

        sliderIncAng.oninput = updateRefraction;
        sliderN1.oninput = updateRefraction;
        sliderN2.oninput = updateRefraction;
        updateRefraction();

        //---------------------------------------------------------
        // 3D Transform Cube Visualizer Logic
        //---------------------------------------------------------
        const canvas = document.getElementById('cube-canvas');
        const ctx = canvas.getContext('2d');
        
        const sliderTx = document.getElementById('slider-tx');
        const sliderRy = document.getElementById('slider-ry');
        const sliderRx = document.getElementById('slider-rx');
        const sliderScale = document.getElementById('slider-scale');

        // Cube vertices
        const vertices = [
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ];

        // Cube edges connections
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], // Back face
            [4, 5], [5, 6], [6, 7], [7, 4], // Front face
            [0, 4], [1, 5], [2, 6], [3, 7]  // Connectors
        ];

        function multiplyMatrixVector(m, v) {
            const out = [0, 0, 0, 0];
            for (let i = 0; i < 4; i++) {
                out[i] = m[i][0]*v[0] + m[i][1]*v[1] + m[i][2]*v[2] + m[i][3]*v[3];
            }
            return out;
        }

        function updateCube() {
            const tx = parseFloat(sliderTx.value);
            const ry = parseFloat(sliderRy.value) * Math.PI / 180;
            const rx = parseFloat(sliderRx.value) * Math.PI / 180;
            const scale = parseFloat(sliderScale.value) / 10;

            document.getElementById('val-tx').innerText = (tx/10).toFixed(1);
            document.getElementById('val-ry').innerText = sliderRy.value + '°';
            document.getElementById('val-rx').innerText = sliderRx.value + '°';
            document.getElementById('val-scale').innerText = scale.toFixed(1);

            // Construct Scale Matrix
            const sMat = [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1]
            ];

            // Construct Rotation Y Matrix
            const cosY = Math.cos(ry);
            const sinY = Math.sin(ry);
            const ryMat = [
                [cosY, 0, sinY, 0],
                [0, 1, 0, 0],
                [-sinY, 0, cosY, 0],
                [0, 0, 0, 1]
            ];

            // Construct Rotation X Matrix
            const cosX = Math.cos(rx);
            const sinX = Math.sin(rx);
            const rxMat = [
                [1, 0, 0, 0],
                [0, cosX, -sinX, 0],
                [0, sinX, cosX, 0],
                [0, 0, 0, 1]
            ];

            // Combine rotations: R = RY * RX
            const rMat = [];
            for (let r = 0; r < 4; r++) {
                rMat[r] = [];
                for (let c = 0; c < 4; c++) {
                    let sum = 0;
                    for (let k = 0; k < 4; k++) {
                        sum += ryMat[r][k] * rxMat[k][c];
                    }
                    rMat[r][c] = sum;
                }
            }

            // Translation (adjust visual offsets, scale division is just for presentation)
            const tMat = [
                [1, 0, 0, tx/20],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ];

            // TRS = T * R * S
            const trsMat = [];
            // Temporary variable for R * S
            const rsMat = [];
            for (let r = 0; r < 4; r++) {
                rsMat[r] = [];
                for (let c = 0; c < 4; c++) {
                    let sum = 0;
                    for (let k = 0; k < 4; k++) {
                        sum += rMat[r][k] * sMat[k][c];
                    }
                    rsMat[r][c] = sum;
                }
            }
            for (let r = 0; r < 4; r++) {
                trsMat[r] = [];
                for (let c = 0; c < 4; c++) {
                    let sum = 0;
                    for (let k = 0; k < 4; k++) {
                        sum += tMat[r][k] * rsMat[k][c];
                    }
                    trsMat[r][c] = sum;
                }
            }

            // Update Matrix UI display
            let mStr = "";
            for (let r = 0; r < 4; r++) {
                mStr += "[ ";
                for (let c = 0; c < 4; c++) {
                    let val = trsMat[r][c];
                    mStr += (val >= 0 ? " " : "") + val.toFixed(2) + "  ";
                }
                mStr = mStr.trim() + " ]\n";
            }
            document.getElementById('matrix-display').innerText = mStr;

            // Render Cube
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw axis guide at center
            ctx.strokeStyle = '#1E2433';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(canvas.width/2, 0);
            ctx.lineTo(canvas.width/2, canvas.height);
            ctx.moveTo(0, canvas.height/2);
            ctx.lineTo(canvas.width, canvas.height/2);
            ctx.stroke();

            // Transform vertices & Project to 2D
            const projected = [];
            const distance = 4; // Camera distance
            const scaleProj = 200; // Screen scaling factor

            for (let i = 0; i < vertices.length; i++) {
                const v = [...vertices[i], 1.0];
                const transformed = multiplyMatrixVector(trsMat, v);
                
                // Camera perspective projection
                // Camera looks down -Z axis, translate along Z by 'distance'
                const z = transformed[2] + distance;
                let x2d = 0;
                let y2d = 0;
                if (Math.abs(z) > 0.001) {
                    x2d = (transformed[0] / z) * scaleProj + canvas.width / 2;
                    y2d = (transformed[1] / z) * scaleProj + canvas.height / 2;
                } else {
                    x2d = transformed[0] * scaleProj + canvas.width / 2;
                    y2d = transformed[1] * scaleProj + canvas.height / 2;
                }
                projected.push([x2d, y2d]);
            }

            // Draw edges
            ctx.strokeStyle = '#4ECDC4';
            ctx.lineWidth = 2.5;
            ctx.shadowBlur = 15;
            ctx.shadowColor = '#4ECDC4';
            
            for (let i = 0; i < edges.length; i++) {
                const p0 = projected[edges[i][0]];
                const p1 = projected[edges[i][1]];
                
                ctx.beginPath();
                ctx.moveTo(p0[0], p0[1]);
                ctx.lineTo(p1[0], p1[1]);
                ctx.stroke();
            }
            
            // Draw vertices as small glowing dots
            ctx.fillStyle = '#FFF';
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#FFF';
            for (let i = 0; i < projected.length; i++) {
                ctx.beginPath();
                ctx.arc(projected[i][0], projected[i][1], 4, 0, 2 * Math.PI);
                ctx.fill();
            }
            ctx.shadowBlur = 0; // Reset
        }

        sliderTx.oninput = updateCube;
        sliderRy.oninput = updateCube;
        sliderRx.oninput = updateCube;
        sliderScale.oninput = updateCube;
        updateCube();
    </script>
</body>
</html>
)HTML";

    file.close();
    std::cout << "Successfully generated " << filepath << "\n";
}

// Function to animate a 3D wireframe cube rotating in the console terminal
void RunTerminalCubeAnimation()
{
    // Define 3D cube vertices
    Vec3f localVertices[8] = {
        {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f}, { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f},
        {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f}, { 1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f}
    };

    // Connections between vertices forming the 12 edges
    std::pair<int, int> edges[12] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Back face
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Front face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Cross edges
    };

    int width = 80;
    int height = 40;
    AsciiRenderer renderer(width, height);

    float angleX = 0.0f;
    float angleY = 0.0f;

    std::cout << "\033[2J"; // Clear screen
    std::cout << "=========================================================\n";
    std::cout << "          KAIROMATH 3D TERMINAL DEMO                     \n";
    std::cout << "=========================================================\n";
    std::cout << "Preparing 3D rotating cube animation in console...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Run for 60 frames (~3 seconds at 20fps)
    for (int frame = 0; frame < 60; ++frame)
    {
        renderer.Clear();

        // 3D rotation quaternion
        angleX += 0.05f;
        angleY += 0.08f;
        Quaternion<float> rotation = RotationAroundX(angleX) * RotationAroundY(angleY);

        // Uniform transform: rotation, scale 8.0, translation centered in viewport
        Transformf transform(Vec3f(0.0f, 0.0f, 0.0f), rotation, Vec3f(8.0f, 8.0f, 8.0f));

        // Let's project vertices into 2D screen coordinates
        std::pair<int, int> projected[8];
        float cameraDistance = 35.0f;

        for (int i = 0; i < 8; ++i)
        {
            Vec3f v = TransformPoint(transform, localVertices[i]);
            
            // Simple camera projection
            // x' = x / (z + d), y' = y / (z + d)
            float depth = v.z + cameraDistance;
            
            // Project and scale to renderer dimensions
            int screenX = static_cast<int>(width / 2 + (v.x / depth) * 60.0f);
            int screenY = static_cast<int>(height / 2 + (v.y / depth) * 35.0f * 0.5f); // Aspect ratio correction
            
            projected[i] = {screenX, screenY};
        }

        // Draw edges
        for (int i = 0; i < 12; ++i)
        {
            auto p0 = projected[edges[i].first];
            auto p1 = projected[edges[i].second];
            renderer.DrawLine(p0.first, p0.second, p1.first, p1.second, '#');
        }

        // Render to stdout
        renderer.Render();
        std::cout << "Rendering frame " << frame + 1 << "/60... Press Ctrl+C to abort.\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int main()
{
    std::cout << "\n=========================================================\n";
    std::cout << "          KAIROMATH STABILIZATION PASS                   \n";
    std::cout << "=========================================================\n\n";

    // 1. Vector operation demo checks
    std::cout << "[Verification] Testing Project() on non-unit axis:\n";
    Vec2f v2(2.0f, 3.0f);
    Vec2f onto2(4.0f, 0.0f);
    Vec2f proj2 = Project(v2, onto2);
    std::cout << "  Project((2, 3), onto (4, 0)) = (" << proj2.x << ", " << proj2.y << ")\n\n";

    std::cout << "[Verification] Testing Refract() near critical angle:\n";
    Vec3f incident = Normalize(Vec3f(0.8f, -0.6f, 0.0f));
    Vec3f normal(0.0f, 1.0f, 0.0f);
    float eta1 = 1.0f / 1.5f; // air to glass
    float eta2 = 1.5f / 1.0f; // glass to air (steep, TIR potential)
    
    Vec3f refr1 = Refract(incident, normal, eta1);
    Vec3f refr2 = Refract(incident, normal, eta2);
    
    std::cout << "  Incident: (" << incident.x << ", " << incident.y << ", " << incident.z << ")\n";
    std::cout << "  Refracted (air->glass): (" << refr1.x << ", " << refr1.y << ", " << refr1.z << ")\n";
    std::cout << "  Refracted (glass->air - TIR expected): (" << refr2.x << ", " << refr2.y << ", " << refr2.z << ")\n\n";

    // 2. Generate the HTML report
    std::string htmlPath = "visual_tests.html";
    GenerateHTMLVisualizer(htmlPath);

    // 3. Animate the 3D cube in terminal
    RunTerminalCubeAnimation();

    std::cout << "\n=========================================================\n";
    std::cout << "  Visual tests generated. Run the following command:\n";
    std::cout << "      open visual_tests.html\n";
    std::cout << "  to interact with the math operations in your browser!\n";
    std::cout << "=========================================================\n\n";

    return 0;
}
