from flask import Flask, render_template_string, request, jsonify, send_file
import speech_recognition as sr
import subprocess
import os
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS

app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

# ---------------- AI Model ----------------
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# ---------------- Helper ----------------
def handle_math(user_input):
    if any(word in user_input.lower() for word in ["add", "subtract", "multiply", "divide", "how many", "grams", "kg", "kilograms"]):
        try:
            numbers = [float(n) for n in re.findall(r'\d+\.?\d*', user_input)]
            answer = 1
            for n in numbers:
                answer *= n
            if "gram" in user_input.lower():
                answer /= 1000
            return f"The answer is {answer} kg."
        except:
            return "I am unable to compute that calculation at the moment."
    return None

# ---------------- AI Reply ----------------
def get_ai_reply(user_input):
    reply = handle_math(user_input)
    if reply:
        return reply

    messages = [
        {"role": "system", "content": "You are a friendly AI assistant."},
        {"role": "user", "content": user_input}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=250,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply_lines = decoded.split("assistant")
    reply = reply_lines[-1].strip() if len(reply_lines) > 1 else decoded.strip()
    return reply


# ---------------- HTML ----------------
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title> AI Chatbot With Voice Assistant </title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --dark: #0f172a;
        --darker: #0a0f1c;
        --light: #f8fafc;
        --gray: #64748b;
        --gray-dark: #334155;
        --success: #22c55e;
        --warning: #f59e0b;
        --error: #ef4444;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, var(--darker), var(--dark));
        color: var(--light);
        height: 100vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .header {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
       
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        z-index: 10;
    }
    
    
    .status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: var(--gray);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--success);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .container {
        display: flex;
        flex: 1;
        overflow: hidden;
    }
    
    .sidebar {
        width: 280px;
        background: rgba(15, 23, 42, 0.7);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        overflow-y: auto;
    }
    
    .sidebar-section {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    
    .sidebar-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .chat-history {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .history-item {
        padding: 0.75rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.875rem;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .chat-header {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
    }
    
    .chat-info h3 {
        font-weight: 600;
        font-size: 1rem;
    }
    
    .chat-info p {
        font-size: 0.875rem;
        color: var(--gray);
    }
    
    .messages-container {
        flex: 1;
        padding: 1.5rem;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }
    
    .message {
        display: flex;
        gap: 0.75rem;
        max-width: 80%;
        animation: fadeIn 0.3s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message.user {
        align-self: flex-end;
        flex-direction: row-reverse;
    }
    
    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        flex-shrink: 0;
    }
    
    .user .message-avatar {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    }
    
    .bot .message-avatar {
        background: linear-gradient(135deg, var(--secondary), #0d9488);
    }
    
    .message-content {
        padding: 0.75rem 1rem;
        border-radius: 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .user .message-content {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        border-bottom-right-radius: 4px;
    }
    
    .bot .message-content {
        background: rgba(255, 255, 255, 0.1);
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .message-time {
        font-size: 0.75rem;
        color: White;
        margin-top: 0.25rem;
        text-align: right;
    }
    
    .bot .message-time {
        text-align: left;
    }
    
    .thinking {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        max-width: 80%;
        align-self: flex-start;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: pulse 2s infinite;
    }
    
    .thinking-dots {
        display: flex;
        gap: 4px;
    }
    
    .thinking-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--gray);
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .input-area {
        padding: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 23, 42, 0.7);
    }
    
    .input-container {
        display: flex;
        gap: 0.75rem;
        align-items: flex-end;
    }
    
    .text-input {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 0.75rem 1.25rem;
        color: var(--light);
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        resize: none;
        max-height: 120px;
        transition: all 0.2s;
    }
    
    .text-input:focus {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    .text-input::placeholder {
        color: var(--gray);
    }
    
    .action-buttons {
        display: flex;
        gap: 0.5rem;
    }
    
    .btn {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 1.25rem;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        color: white;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.1);
        color: var(--light);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    .btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none !important;
        box-shadow: none !important;
    }
    
    .audio-player {
        margin-top: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: var(--gray);
    }
    
    .audio-control {
        background: none;
        border: none;
        color: var(--primary);
        cursor: pointer;
        font-size: 1rem;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .sidebar {
            display: none;
        }
        
        .message {
            max-width: 90%;
        }
    }
</style>
</head>
<body>
    <div class="header">
       
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">Quick Actions</div>
                <div class="history-item">Wikipedia</div>
            
                <div class="history-item">Explain About History</div>
                <div class="history-item">Math calculation</div>
 
                <div class="history-item">Content & Writing</div>
                <div class="history-item">Programming Help</div>
                <div class="history-item">Learning & Education</div>
                <div class="history-item">Suggestions</div>
               
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Capabilities</div>
                <div class="history-item">Voice & Text Input</div>
                <div class="history-item">Mathematical Calculations</div>
                <div class="history-item">General Knowledge</div>
                <div class="history-item">Text-to-Speech</div>
            </div>
        </div>
        
        <div class="chat-area">
            <div class="chat-header">
                <div class="chat-avatar">AI</div>
                <div class="chat-info">
                    <h3>AI Chat Assistant</h3>
                    <p>Always ready to help</p>
                </div>
            </div>
            
            <div class="messages-container" id="messagesContainer">
                <div class="message bot">
                    <div class="message-avatar">AI</div>
                    <div class="message-content">
                        Hello! I'm your AI assistant. How can I help you today?
                        <div class="message-time">Just now</div>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-container">
                    <textarea class="text-input" id="userInput" placeholder="Type your message or use voice input..." rows="1"></textarea>
                    <div class="action-buttons">
                        <button class="btn btn-secondary" id="voiceBtn" title="Voice Input">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 14C13.66 14 14.99 12.66 14.99 11L15 5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V11C9 12.66 10.34 14 12 14ZM17.3 11C17.3 14 14.76 16.1 12 16.1C9.24 16.1 6.7 14 6.7 11H5C5 14.41 7.72 17.23 11 17.72V21H13V17.72C16.28 17.24 19 14.42 19 11H17.3Z" fill="currentColor"/>
                            </svg>
                        </button>
                        <button class="btn btn-primary" id="sendBtn" title="Send Message">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="audio-player" id="audioPlayer" style="display: none;">
                    <button class="audio-control" id="playAudioBtn">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M8 5v14l11-7z" fill="currentColor"/>
                        </svg>
                    </button>
                    <span>Play AI response</span>
                </div>
            </div>
        </div>
    </div>
    
    <audio id="audioElement" style="display: none;"></audio>

    <script>
        const messagesContainer = document.getElementById("messagesContainer");
        const userInput = document.getElementById("userInput");
        const sendBtn = document.getElementById("sendBtn");
        const voiceBtn = document.getElementById("voiceBtn");
        const audioPlayer = document.getElementById("audioPlayer");
        const playAudioBtn = document.getElementById("playAudioBtn");
        const audioElement = document.getElementById("audioElement");
        
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let currentAudioUrl = null;
        
        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Send message on Enter (but allow Shift+Enter for new line)
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        sendBtn.onclick = sendMessage;
        
        function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Show thinking indicator
            showThinking();
            
            // Send to backend
            fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message})
            })
            .then(res => res.json())
            .then(data => {
                hideThinking();
                addMessage(data.reply, 'bot');
                
                // Show audio player if we have audio
                if (data.audio_url) {
                    currentAudioUrl = data.audio_url;
                    audioPlayer.style.display = 'flex';
                }
            })
            .catch(err => {
                hideThinking();
                addMessage("Sorry, I encountered an error. Please try again.", 'bot');
                console.error(err);
            });
        }
        
        function addMessage(text, sender) {
            const messageDiv = document.createElement("div");
            messageDiv.className = `message ${sender}`;
            
            const avatarDiv = document.createElement("div");
            avatarDiv.className = "message-avatar";
            avatarDiv.textContent = sender === 'user' ? 'You' : 'AI';
            
            const contentDiv = document.createElement("div");
            contentDiv.className = "message-content";
            contentDiv.textContent = text;
            
            const timeDiv = document.createElement("div");
            timeDiv.className = "message-time";
            timeDiv.textContent = getCurrentTime();
            
            contentDiv.appendChild(timeDiv);
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
            
            messagesContainer.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function showThinking() {
            const thinkingDiv = document.createElement("div");
            thinkingDiv.className = "thinking";
            thinkingDiv.id = "thinkingIndicator";
            
            const avatarDiv = document.createElement("div");
            avatarDiv.className = "message-avatar";
            avatarDiv.textContent = "AI";
            
            const contentDiv = document.createElement("div");
            contentDiv.className = "thinking-dots";
            
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement("div");
                dot.className = "thinking-dot";
                contentDiv.appendChild(dot);
            }
            
            thinkingDiv.appendChild(avatarDiv);
            thinkingDiv.appendChild(contentDiv);
            
            messagesContainer.appendChild(thinkingDiv);
            scrollToBottom();
        }
        
        function hideThinking() {
            const thinkingIndicator = document.getElementById("thinkingIndicator");
            if (thinkingIndicator) {
                thinkingIndicator.remove();
            }
        }
        
        function getCurrentTime() {
            const now = new Date();
            return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        
        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Voice recording functionality
        voiceBtn.onclick = toggleRecording;
        
        async function toggleRecording() {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        noiseSuppression: true,
                        echoCancellation: true,
                        autoGainControl: true
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream, { 
                    mimeType: "audio/webm;codecs=opus" 
                });
                
                audioChunks = [];
                mediaRecorder.ondataavailable = e => {
                    if (e.data.size > 0) audioChunks.push(e.data);
                };
                
                mediaRecorder.start();
                isRecording = true;
                voiceBtn.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor"/>
                    </svg>
                `;
                voiceBtn.style.background = "linear-gradient(135deg, #ef4444, #dc2626)";
                
                addMessage("Recording... Speak now", 'user');
            } catch (err) {
                console.error("Error accessing microphone:", err);
                addMessage("Microphone access denied. Please allow microphone permissions.", 'bot');
            }
        }
        
        async function stopRecording() {
            if (!mediaRecorder) return;
            
            mediaRecorder.stop();
            isRecording = false;
            
            voiceBtn.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 14C13.66 14 14.99 12.66 14.99 11L15 5C15 3.34 13.66 2 12 2C10.34 2 9 3.34 9 5V11C9 12.66 10.34 14 12 14ZM17.3 11C17.3 14 14.76 16.1 12 16.1C9.24 16.1 6.7 14 6.7 11H5C5 14.41 7.72 17.23 11 17.72V21H13V17.72C16.28 17.24 19 14.42 19 11H17.3Z" fill="currentColor"/>
                </svg>
            `;
            voiceBtn.style.background = "";
            
            // Wait for recording to stop
            await new Promise(resolve => mediaRecorder.onstop = resolve);
            
            const blob = new Blob(audioChunks, { type: "audio/webm" });
            const formData = new FormData();
            formData.append("audio", blob, "voice.webm");
            
            // Show thinking indicator
            showThinking();
            
            // Send to backend
            fetch("/voice_chat", { 
                method: "POST", 
                body: formData 
            })
            .then(res => res.json())
            .then(data => {
                hideThinking();
                
                // Update the last user message with transcribed text
                const userMessages = document.querySelectorAll('.message.user');
                if (userMessages.length > 0) {
                    const lastUserMessage = userMessages[userMessages.length - 1];
                    const contentDiv = lastUserMessage.querySelector('.message-content');
                    if (contentDiv) {
                        contentDiv.textContent = data.text;
                        const timeDiv = document.createElement("div");
                        timeDiv.className = "message-time";
                        timeDiv.textContent = getCurrentTime();
                        contentDiv.appendChild(timeDiv);
                    }
                }
                
                addMessage(data.reply, 'bot');
                
                // Show audio player if we have audio
                if (data.audio_url) {
                    currentAudioUrl = data.audio_url;
                    audioPlayer.style.display = 'flex';
                }
            })
            .catch(err => {
                hideThinking();
                addMessage("Sorry, I encountered an error processing your voice message.", 'bot');
                console.error(err);
            });
            
            // Stop all tracks
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        // Audio playback
        playAudioBtn.onclick = function() {
            if (currentAudioUrl) {
                audioElement.src = currentAudioUrl;
                audioElement.play();
            }
        };
        
        // Initialize with a welcome message
        window.onload = function() {
            scrollToBottom();
        };
    </script>
</body>
</html>
    """)


# ---------------- Text Chat ----------------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    reply = get_ai_reply(user_input)
    
    # Generate TTS audio
    tts_path = f"uploads/{int(time.time())}_reply.mp3"
    try:
        tts = gTTS(reply)
        tts.save(tts_path)
        audio_url = f"/get_audio/{os.path.basename(tts_path)}"
    except Exception as e:
        print(f"TTS Error: {e}")
        audio_url = None
    
    return jsonify({
        "reply": reply,
        "audio_url": audio_url
    })


# ---------------- Voice Chat ----------------
@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    try:
        file = request.files["audio"]
        timestamp = str(int(time.time()))
        webm_path = f"uploads/{timestamp}.webm"
        wav_path = f"uploads/{timestamp}.wav"
        tts_path = f"uploads/{timestamp}_reply.mp3"

        file.save(webm_path)
        cmd = ["ffmpeg", "-y", "-i", webm_path, "-ar", "44100", "-ac", "1", wav_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        reply = get_ai_reply(text)
        
        # Generate TTS audio
        try:
            tts = gTTS(reply)
            tts.save(tts_path)
            audio_url = f"/get_audio/{os.path.basename(tts_path)}"
        except Exception as e:
            print(f"TTS Error: {e}")
            audio_url = None

        # Cleanup temporary files
        os.remove(webm_path)
        os.remove(wav_path)

        return jsonify({
            "text": text,
            "reply": reply,
            "audio_url": audio_url
        })
    except sr.UnknownValueError:
        return jsonify({"text": "Could not understand audio", "reply": "Sorry, I couldn't understand what you said. Please try again."})
    except Exception as e:
        return jsonify({"text": "Error", "reply": f"Sorry, I encountered an error: {str(e)}"})


# ---------------- Audio File Serving ----------------
@app.route("/get_audio/<filename>")
def get_audio(filename):
    return send_file(f"uploads/{filename}", as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)