

#python-----------------------------------


# app.py (Updated)
from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """ This new route handles the chat logic. """
    user_message = request.json.get('message', '').lower()
    
    # Simulate a small delay to feel like a real bot is "thinking"
    time.sleep(0.5)
    
    bot_response = "I'm not sure how to respond to that. Try asking about 'themes' or 'creator'."
    
    # Simple keyword-based responses
    if 'hello' in user_message or 'hi' in user_message:
        bot_response = "Hello there! How can I help you today?"
    elif 'theme' in user_message or 'color' in user_message:
        bot_response = "You can change the color theme using the palette icon in the top-right corner!"
    elif 'creator' in user_message or 'who made you' in user_message:
        bot_response = "I was created from a creative idea and brought to life with Python and code."
    elif 'awesome' in user_message or 'cool' in user_message or 'love this' in user_message:
        bot_response = "Thank you! I'm glad you like it."
        
    return jsonify({'reply': bot_response})

if __name__ == '__main__':
    app.run(debug=True)


#  index.html---------------------------------------------------



<!-- templates/index.html (Updated) -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Volcanic Flow UI</title>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Alex+Brush&display=swap" rel="stylesheet">
    
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <canvas id="background-canvas"></canvas>

    <!-- New Theme Switcher Button -->
    <div id="theme-switcher" title="Change Theme">
        <!-- A simple SVG icon for a color palette -->
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9c.83 0 1.5-.67 1.5-1.5 0-.39-.15-.74-.39-1.01-.23-.26-.38-.61-.38-.99 0-.83.67-1.5 1.5-1.5H16c2.76 0 5-2.24 5-5 0-4.42-4.03-8-9-8zm-5.5 9c-.83 0-1.5-.67-1.5-1.5S5.67 9 6.5 9 8 9.67 8 10.5 7.33 12 6.5 12zm3-4c-.83 0-1.5-.67-1.5-1.5S8.67 5 9.5 5s1.5.67 1.5 1.5S10.33 8 9.5 8zm5 0c-.83 0-1.5-.67-1.5-1.5S13.67 5 14.5 5s1.5.67 1.5 1.5S15.33 8 14.5 8zm3 4c-.83 0-1.5-.67-1.5-1.5S16.67 9 17.5 9s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>
        </svg>
    </div>

    <div class="ui-container">
        <input type="text" id="chat-input" placeholder="Ask anything...">
        <div id="feedback-message"></div>
    </div>

    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>


# ----------------------------------------------------
  # css


/* static/style.css (Updated) */

/* --- Variable Definitions for Themes --- */
:root {
    /* Default Theme (as before) */
    --background-color: #0d1018;
    --line-color: rgba(230, 230, 230, 0.6);
    --accent-color: #EDFF00;
    --text-color: #ffffff;
    --selection-pink: #ff69b4;
    --feedback-font: 'Alex Brush', cursive;
}

body[data-theme='ocean'] {
    --background-color: #021024;
    --line-color: rgba(122, 196, 222, 0.6);
    --accent-color: #72cce3;
}

body[data-theme='fire'] {
    --background-color: #1a0000;
    --line-color: rgba(255, 170, 0, 0.6);
    --accent-color: #ff4500;
}

body[data-theme='space'] {
    --background-color: #000000;
    --line-color: rgba(200, 160, 220, 0.6);
    --accent-color: #dda0dd;
}


/* --- Global Styles --- */
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 0;
    font-family: sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
    overflow: hidden;
    transition: background-color 0.5s ease; /* Smooth transition for theme change */
}
::selection { background-color: var(--selection-pink); color: var(--background-color); }

/* --- Animated Background Canvas --- */
#background-canvas {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}

/* --- Theme Switcher Button --- */
#theme-switcher {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 50px;
    height: 50px;
    background-color: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 10;
}
#theme-switcher:hover {
    transform: scale(1.1) rotate(15deg);
    background-color: rgba(255, 255, 255, 0.2);
}
#theme-switcher svg {
    width: 28px;
    height: 28px;
    color: var(--accent-color);
    transition: color 0.5s ease;
}


/* --- UI Container & Load-in Animations --- */
.ui-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    padding: 20px;
}

/* Animation Keyframes */
@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* --- Transparent Chat/Search Bar --- */
#chat-input {
    width: 100%;
    max-width: 600px;
    padding: 18px 25px;
    font-size: 1.2em;
    color: var(--text-color);
    background-color: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 50px;
    outline: none;
    transition: all 0.3s ease;
    /* Apply animation */
    animation: fade-in-up 0.8s ease-out 0.2s forwards;
    opacity: 0; /* Start hidden for animation */
}
#chat-input::placeholder { color: rgba(255, 255, 255, 0.5); }
#chat-input:focus {
    border-color: var(--accent-color);
    box-shadow: 0 0 15px var(--accent-color);
}

/* --- Feedback/Notification Area --- */
#feedback-message {
    margin-top: 25px;
    font-family: var(--feedback-font);
    font-style: italic;
    font-size: 2.2em;
    color: var(--accent-color);
    text-shadow: 0 0 8px var(--accent-color);
    height: 50px;
    transition: opacity 0.5s ease, color 0.5s ease;
    text-align: center;
    /* Apply animation with a delay */
    animation: fade-in-up 0.8s ease-out 0.4s forwards;
    opacity: 0; /* Start hidden for animation */
}



#------------------------------------------------

#JS


// static/script.js (Updated)
document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const canvas = document.getElementById('background-canvas');
    const ctx = canvas.getContext('2d');
    const themeSwitcher = document.getElementById('theme-switcher');
    const chatInput = document.getElementById('chat-input');
    const feedbackMessage = document.getElementById('feedback-message');

    // --- State Variables ---
    let width, height;
    const particles = [];
    const particleCount = 500;
    const noiseScale = 0.003;
    const particleSpeed = 0.5;
    const lineOpacity = 0.05;
    let feedbackTimeout;
    const themes = ['default', 'ocean', 'fire', 'space'];
    let currentThemeIndex = 0;

    // --- Mouse Interaction Variables ---
    const mouse = {
        x: null,
        y: null,
        radius: 100 // The radius of repulsion from the cursor
    };

    window.addEventListener('mousemove', (event) => {
        mouse.x = event.x;
        mouse.y = event.y;
    });
    window.addEventListener('mouseout', () => { // Reset when mouse leaves window
        mouse.x = null;
        mouse.y = null;
    });

    // --- Noise Generator (same as before) ---
    const noise = (() => { /* ... Perlin noise code is unchanged, keeping it for brevity ... */
        let p = new Uint8Array(512); for (let i=0; i < 256; i++) p[i] = p[i+256] = Math.floor(Math.random()*256); function fade(t) { return t*t*t*(t*(t*6-15)+10); } function lerp(t,a,b) { return a+t*(b-a); } function grad(hash,x,y,z) { let h=hash&15,u=h<8?x:y,v=h<4?y:h==12||h==14?x:z; return ((h&1)==0?u:-u)+((h&2)==0?v:-v); } return { noise: function(x,y,z) { let X=Math.floor(x)&255,Y=Math.floor(y)&255,Z=Math.floor(z)&255; x-=Math.floor(x);y-=Math.floor(y);z-=Math.floor(z); let u=fade(x),v=fade(y),w=fade(z); let A=p[X]+Y,AA=p[A]+Z,AB=p[A+1]+Z,B=p[X+1]+Y,BA=p[B]+Z,BB=p[B+1]+Z; return lerp(w,lerp(v,lerp(u,grad(p[AA],x,y,z),grad(p[BA],x-1,y,z)),lerp(u,grad(p[AB],x,y-1,z),grad(p[BB],x-1,y-1,z))),lerp(v,lerp(u,grad(p[AA+1],x,y,z-1),grad(p[BA+1],x-1,y,z-1)),lerp(u,grad(p[AB+1],x,y-1,z-1),grad(p[BB+1],x-1,y-1,z-1))));}}
    })();

    // --- Particle Class with Mouse Repulsion ---
    class Particle {
        constructor() { this.x = Math.random() * width; this.y = Math.random() * height; }
        update(time) {
            const angle = noise.noise(this.x * noiseScale, this.y * noiseScale, time * 0.0001) * Math.PI * 2;
            let vx = Math.cos(angle) * particleSpeed;
            let vy = Math.sin(angle) * particleSpeed;

            // Mouse Repulsion Logic
            if (mouse.x != null) {
                const dx = this.x - mouse.x;
                const dy = this.y - mouse.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < mouse.radius) {
                    const force = (mouse.radius - distance) / mouse.radius;
                    vx += (dx / distance) * force * 2; // Push away from mouse
                    vy += (dy / distance) * force * 2;
                }
            }

            this.x += vx; this.y += vy;
            if (this.x > width) this.x = 0; if (this.x < 0) this.x = width;
            if (this.y > height) this.y = 0; if (this.y < 0) this.y = height;
        }
        draw(ctx) { ctx.beginPath(); ctx.arc(this.x, this.y, 1, 0, Math.PI * 2); ctx.fill(); }
    }
    
    // --- Canvas Setup and Animation Loop ---
    function setup() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
        updateCanvasColors();
        particles.length = 0;
        for (let i = 0; i < particleCount; i++) { particles.push(new Particle()); }
    }

    function animate(time) {
        const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--background-color').trim();
        // Convert hex to rgb for rgba
        const r = parseInt(bgColor.slice(1, 3), 16);
        const g = parseInt(bgColor.slice(3, 5), 16);
        const b = parseInt(bgColor.slice(5, 7), 16);

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${lineOpacity})`;
        ctx.fillRect(0, 0, width, height);
        
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--line-color').trim();

        particles.forEach(p => { p.update(time); p.draw(ctx); });
        requestAnimationFrame(animate);
    }

    function updateCanvasColors() {
        const currentLineColor = getComputedStyle(document.documentElement).getPropertyValue('--line-color').trim();
        ctx.fillStyle = currentLineColor;
    }

    // --- Event Listeners ---
    themeSwitcher.addEventListener('click', () => {
        currentThemeIndex = (currentThemeIndex + 1) % themes.length;
        document.body.dataset.theme = themes[currentThemeIndex];
        updateCanvasColors(); // Important: Update canvas colors on theme change
    });
    
    // --- Updated Chat Input Logic ---
    chatInput.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter' && chatInput.value.trim() !== '') {
            e.preventDefault();
            const userText = chatInput.value;
            chatInput.value = '';

            // Show an immediate "thinking" feedback
            showFeedback("Thinking...");

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userText })
                });

                if (!response.ok) { throw new Error('Network response was not ok.'); }
                
                const data = await response.json();
                showFeedback(data.reply, 8000); // Show bot reply for 8 seconds
            } catch (error) {
                console.error('Error:', error);
                showFeedback("Sorry, something went wrong.", 5000);
            }
        }
    });

    function showFeedback(message, duration = 4000) {
        feedbackMessage.textContent = message;
        feedbackMessage.style.opacity = '1';
        clearTimeout(feedbackTimeout);
        feedbackTimeout = setTimeout(() => {
            feedbackMessage.style.opacity = '0';
        }, duration);
    }

    // --- Initial Run ---
    setup();
    requestAnimationFrame(animate);
    window.addEventListener('resize', setup);
});


#THIS ONE IS UPDATE TO THE PREVIOUS CODE 3 FILE , I M NAMING IT CODE 3A , IT JUST ADDS MORE DYNAMIC EFFECTS 

