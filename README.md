# 🚀 Puchsach MCP Server

*An intelligent analysis and assistance server powered by FastMCP and local LLM integration* 🤖✨

---

## 📖 Overview

This *Modular Chatbot Platform (MCP) server* helps you:

- ✅ Fact-check statements with web context  
- 🎭 Analyze message tone and style  
- 🌐 Translate messages between languages  
- ⚠ Detect scam or phishing content  
- 🤝 Synthesize answers from web results using local LLM reasoning  

Built with async Python, HTTP clients, HTML extraction, and Ollama’s LLaMA 3.2 model for smart, conversational assistance.

---

## ✨ Features

| Feature                       | Description                                                       |
|------------------------------|-------------------------------------------------------------------|
| 🔐 Secure Authentication      | Custom Bearer Token with RSA Key generation                       |
| 📄 HTML to Markdown           | Clean extraction using readabilipy & markdownify             |
| 🕵 Fact-Checking & Scam Detection | Detailed reports on truthfulness & phishing risks                |
| 💬 Message Tone & Translation | Emotional tone analysis & natural language translation            |
| 🌍 DuckDuckGo Web Scraping    | Lightweight, programmatic web search scraping                     |
| 🧠 LLM Reasoning              | Advanced analysis & summarization via local Ollama LLaMA API     |
| ⚡ Async Modular Server       | Fast, extensible tools using FastMCP                              |

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.9+  
- [FastMCP](https://github.com/your-fork/fastmcp) installed  
- A .env file with:

```env
AUTH_TOKEN=your_secret_token_here
MY_NUMBER=your_identifier_here
OLLAMA_API_URL=http://localhost:11434/v1/chat/completions  # optional
📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/puchsach.git
cd puchsach
pip install -r requirements.txt
▶ Running the Server
bash
Copy
Edit
python main.py
Access the server at:
http://0.0.0.0:8086

🛠 Usage
Available Tools
Tool Name	Purpose
fact_checker_online	Verify truthfulness of statements with context
message_tone_checker_o	Analyze emotional & stylistic tone of text messages
message_translator_o	Translate messages naturally preserving tone
scam_detector_o	Detect phishing or scam likelihood
whatsapp_search_engine_online	Synthesize answers from web pages & search results

🧑‍💻 Technical Details
HTTP Requests: Async via httpx

Content Extraction: readabilipy + markdownify

Web Scraping: BeautifulSoup

LLM Integration: Ollama LLaMA 3.2 for advanced NLP tasks

Auth: Custom Bearer token with RSA key pair

Server: Async modular tools using FastMCP
