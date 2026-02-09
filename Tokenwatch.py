#!/usr/bin/env python3
"""
TokenWatch - LLM Router for Agents
Self-contained single-file router with dashboard
"""

import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
import ssl

# Configuration - Read from environment or use defaults
PROVIDERS = {
    "anthropic": {
        "name": "Anthropic",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "base_url": "https://api.anthropic.com/v1",
        "models": {
            "claude-opus-4-5": {"input": 15.0, "output": 75.0, "limit": 30000},
            "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "limit": 40000},
            "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "limit": 50000}
        },
        "priority": 1
    },
    "kimi": {
        "name": "Kimi",
        "api_key": os.getenv("KIMI_API_KEY", ""),
        "base_url": "https://api.moonshot.cn/v1",
        "models": {
            "kimi-k2.5": {"input": 2.0, "output": 8.0, "limit": 60000},
            "kimi-k1.5": {"input": 1.0, "output": 4.0, "limit": 60000}
        },
        "priority": 2
    },
    "openai": {
        "name": "OpenAI",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": "https://api.openai.com/v1",
        "models": {
            "gpt-4o": {"input": 2.50, "output": 10.0, "limit": 10000},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60, "limit": 20000}
        },
        "priority": 3
    }
}

# In-memory tracking
usage_windows: Dict[str, List[dict]] = {}
request_count = 0
start_time = time.time()

# Database path
DB_PATH = os.path.expanduser("~/.openclaw/tokenwatch.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT,
        model TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        total_tokens INTEGER,
        cost REAL,
        timestamp REAL,
        task_complexity TEXT,
        response_time_ms INTEGER
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT,
        error TEXT,
        timestamp REAL
    )''')
    conn.commit()
    conn.close()

def log_usage(provider: str, model: str, input_tokens: int, output_tokens: int, 
              cost: float, complexity: str, response_time_ms: int):
    """Log usage to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO usage 
                 (provider, model, input_tokens, output_tokens, total_tokens, cost, timestamp, task_complexity, response_time_ms)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (provider, model, input_tokens, output_tokens, input_tokens + output_tokens,
               cost, time.time(), complexity, response_time_ms))
    conn.commit()
    conn.close()

def log_error(provider: str, error: str):
    """Log error to database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO errors (provider, error, timestamp) VALUES (?, ?, ?)',
              (provider, error, time.time()))
    conn.commit()
    conn.close()

def get_db_stats(hours: int = 24) -> Dict:
    """Get usage statistics from database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    since = time.time() - (hours * 3600)
    
    # Total stats
    c.execute('''SELECT COUNT(*), SUM(total_tokens), SUM(cost), AVG(response_time_ms)
                 FROM usage WHERE timestamp > ?''', (since,))
    total = c.fetchone()
    
    # By provider
    c.execute('''SELECT provider, COUNT(*), SUM(total_tokens), SUM(cost)
                 FROM usage WHERE timestamp > ? GROUP BY provider''', (since,))
    by_provider = c.fetchall()
    
    # Recent errors
    c.execute('''SELECT provider, error, timestamp FROM errors 
                 WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 10''', (since,))
    errors = c.fetchall()
    
    conn.close()
    
    return {
        "total_requests": total[0] or 0,
        "total_tokens": total[1] or 0,
        "total_cost": total[2] or 0,
        "avg_response_ms": total[3] or 0,
        "by_provider": by_provider,
        "errors": errors
    }

def get_provider_usage(provider: str, minutes: int = 1) -> int:
    """Get token usage for provider in last N minutes"""
    now = time.time()
    window_start = now - (minutes * 60)
    
    if provider not in usage_windows:
        usage_windows[provider] = []
    
    # Clean old entries
    usage_windows[provider] = [u for u in usage_windows[provider] if u["time"] > window_start]
    
    return sum(u["tokens"] for u in usage_windows[provider])

def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD"""
    if provider in PROVIDERS and model in PROVIDERS[provider]["models"]:
        pricing = PROVIDERS[provider]["models"][model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)
    return 0.0

def select_provider(complexity: str = "complex", preferred_model: str = None) -> tuple:
    """
    Select best available provider based on rate limits and cost
    Returns: (provider_key, model_name, reason)
    """
    # Sort providers by priority
    sorted_providers = sorted(PROVIDERS.items(), 
                              key=lambda x: x[1]["priority"])
    
    # For simple tasks, prefer cheaper providers
    if complexity == "simple":
        sorted_providers = sorted(PROVIDERS.items(),
                                  key=lambda x: min(m["input"] for m in x[1]["models"].values()))
    
    for provider_key, provider in sorted_providers:
        if not provider["api_key"]:
            continue
        
        # Get primary model or requested model
        if preferred_model and preferred_model in provider["models"]:
            model = preferred_model
        else:
            model = list(provider["models"].keys())[0]
        
        limit = provider["models"][model]["limit"]
        current = get_provider_usage(provider_key)
        
        # Check if under 80% capacity
        if current < (limit * 0.8):
            reason = f"Under limit ({current}/{limit} tokens/min)"
            return (provider_key, model, reason)
    
    # All providers at limit - pick one with lowest current usage
    best_provider = None
    best_usage = float('inf')
    
    for provider_key, provider in sorted_providers:
        if not provider["api_key"]:
            continue
        current = get_provider_usage(provider_key)
        if current < best_usage:
            best_usage = current
            best_provider = provider_key
    
    if best_provider:
        model = list(PROVIDERS[best_provider]["models"].keys())[0]
        return (best_provider, model, "All at limit - using least loaded")
    
    return (None, None, "No providers available")

def make_http_request(url: str, headers: dict, data: dict, timeout: int = 60) -> tuple:
    """Make HTTP request and return (status_code, response_body, error)"""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={**headers, 'Content-Type': 'application/json'},
            method='POST'
        )
        
        # SSL context that allows us to proceed even with cert issues
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
            return (response.status, response.read().decode('utf-8'), None)
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode('utf-8'), str(e))
    except Exception as e:
        return (0, None, str(e))

def route_anthropic(messages: list, model: str, max_tokens: int = 4096) -> dict:
    """Route request to Anthropic"""
    provider = PROVIDERS["anthropic"]
    
    # Convert to Anthropic format
    anthropic_messages = []
    system_msg = None
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content")
        else:
            anthropic_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
    
    # Map model names
    model_map = {
        "claude-opus-4-5": "claude-3-opus-20240229",
        "claude-sonnet-4-5": "claude-3-5-sonnet-20241022",
        "claude-haiku-4-5": "claude-3-5-haiku-20241022"
    }
    anthropic_model = model_map.get(model, "claude-3-opus-20240229")
    
    payload = {
        "model": anthropic_model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages
    }
    if system_msg:
        payload["system"] = system_msg
    
    start = time.time()
    status, body, error = make_http_request(
        f"{provider['base_url']}/messages",
        {"x-api-key": provider["api_key"], "anthropic-version": "2023-06-01"},
        payload
    )
    response_time = int((time.time() - start) * 1000)
    
    if error:
        log_error("anthropic", error)
        return {"error": error, "provider": "anthropic"}
    
    data = json.loads(body)
    
    # Track usage
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = calculate_cost("anthropic", model, input_tokens, output_tokens)
    
    # Update in-memory window
    if "anthropic" not in usage_windows:
        usage_windows["anthropic"] = []
    usage_windows["anthropic"].append({
        "time": time.time(),
        "tokens": input_tokens + output_tokens
    })
    
    return {
        "id": data.get("id", "anthropic-123"),
        "object": "chat.completion",
        "provider": "anthropic",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"][0]["text"] if data.get("content") else ""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost
        },
        "response_time_ms": response_time
    }

def route_kimi(messages: list, model: str, max_tokens: int = 4096) -> dict:
    """Route request to Kimi"""
    provider = PROVIDERS["kimi"]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    start = time.time()
    status, body, error = make_http_request(
        f"{provider['base_url']}/chat/completions",
        {"Authorization": f"Bearer {provider['api_key']}"},
        payload
    )
    response_time = int((time.time() - start) * 1000)
    
    if error:
        log_error("kimi", error)
        return {"error": error, "provider": "kimi"}
    
    data = json.loads(body)
    
    # Track usage
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = calculate_cost("kimi", model, input_tokens, output_tokens)
    
    # Update in-memory window
    if "kimi" not in usage_windows:
        usage_windows["kimi"] = []
    usage_windows["kimi"].append({
        "time": time.time(),
        "tokens": input_tokens + output_tokens
    })
    
    data["provider"] = "kimi"
    data["usage"]["cost_usd"] = cost
    data["response_time_ms"] = response_time
    return data

def route_openai(messages: list, model: str, max_tokens: int = 4096) -> dict:
    """Route request to OpenAI"""
    provider = PROVIDERS["openai"]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    start = time.time()
    status, body, error = make_http_request(
        f"{provider['base_url']}/chat/completions",
        {"Authorization": f"Bearer {provider['api_key']}"},
        payload
    )
    response_time = int((time.time() - start) * 1000)
    
    if error:
        log_error("openai", error)
        return {"error": error, "provider": "openai"}
    
    data = json.loads(body)
    
    # Track usage
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = calculate_cost("openai", model, input_tokens, output_tokens)
    
    # Update in-memory window
    if "openai" not in usage_windows:
        usage_windows["openai"] = []
    usage_windows["openai"].append({
        "time": time.time(),
        "tokens": input_tokens + output_tokens
    })
    
    data["provider"] = "openai"
    data["usage"]["cost_usd"] = cost
    data["response_time_ms"] = response_time
    return data

def chat_completion(request_data: dict) -> dict:
    """Main routing logic for chat completions"""
    global request_count
    request_count += 1
    
    messages = request_data.get("messages", [])
    complexity = request_data.get("complexity", "complex")
    preferred_model = request_data.get("model")
    max_tokens = request_data.get("max_tokens", 4096)
    
    # Select provider
    provider, model, reason = select_provider(complexity, preferred_model)
    
    if not provider:
        return {
            "error": "All providers at rate limit. Please try again in 60 seconds.",
            "type": "rate_limit_exceeded"
        }
    
    # Route to selected provider
    start = time.time()
    
    if provider == "anthropic":
        response = route_anthropic(messages, model, max_tokens)
    elif provider == "kimi":
        response = route_kimi(messages, model, max_tokens)
    elif provider == "openai":
        response = route_openai(messages, model, max_tokens)
    else:
        return {"error": f"Unknown provider: {provider}"}
    
    # Add routing info
    response["routing"] = {
        "provider": provider,
        "model": model,
        "reason": reason,
        "complexity": complexity
    }
    
    return response

# HTTP Request Handler
class TokenWatchHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default logging
        pass
    
    def do_GET(self):
        """Handle GET requests - Dashboard and API"""
        if self.path == "/" or self.path == "/dashboard":
            self.send_dashboard()
        elif self.path == "/api/stats":
            self.send_stats()
        elif self.path == "/api/limits":
            self.send_limits()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests - Chat completions"""
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(post_data.decode('utf-8'))
                response = chat_completion(request_data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_dashboard(self):
        """Send HTML dashboard"""
        stats = get_db_stats(24)
        
        # Get current rate limits
        limits_html = ""
        for provider_key, provider in PROVIDERS.items():
            if not provider["api_key"]:
                continue
            current = get_provider_usage(provider_key)
            model = list(provider["models"].keys())[0]
            limit = provider["models"][model]["limit"]
            percent = (current / limit) * 100
            
            color = "#16c79a" if percent < 50 else "#f9a825" if percent < 80 else "#e94560"
            
            limits_html += f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>{provider['name']}</span>
                    <span>{int(current)} / {limit} tokens/min ({percent:.1f}%)</span>
                </div>
                <div style="height: 20px; background: #0f3460; border-radius: 10px; overflow: hidden;">
                    <div style="height: 100%; width: {percent}%; background: {color}; transition: width 0.3s;"></div>
                </div>
            </div>
            """
        
        # Provider stats table
        provider_rows = ""
        for row in stats["by_provider"]:
            provider_rows += f"""
            <tr>
                <td>{row[0]}</td>
                <td>{row[1]}</td>
                <td>{row[2] or 0:,}</td>
                <td>${row[3] or 0:.4f}</td>
            </tr>
            """
        
        uptime = int(time.time() - start_time)
        uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TokenWatch Dashboard</title>
            <meta http-equiv="refresh" content="5">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                       padding: 20px; background: #0a0a0f; color: #e0e0e0; max-width: 1200px; margin: 0 auto; }}
                .card {{ background: #16161f; padding: 20px; margin: 15px 0; border-radius: 12px; 
                        border: 1px solid #2a2a3a; }}
                h1 {{ color: #00d4ff; margin: 0; }}
                h2 {{ color: #a0a0b0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-top: 0; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
                .stat {{ text-align: center; }}
                .stat-value {{ font-size: 32px; font-weight: bold; color: #00d4ff; }}
                .stat-label {{ color: #808090; font-size: 12px; margin-top: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #2a2a3a; }}
                th {{ color: #808090; font-weight: 500; }}
                .endpoint {{ background: #0f1419; padding: 15px; border-radius: 8px; font-family: monospace; 
                           font-size: 13px; margin-top: 10px; border: 1px solid #2a2a3a; }}
                .status {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }}
                .status.ok {{ background: #16c79a; }}
                .status.warn {{ background: #f9a825; }}
                .status.error {{ background: #e94560; }}
            </style>
        </head>
        <body>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h1>TokenWatch</h1>
                <div style="color: #808090; font-size: 14px;">
                    <span class="status ok"></span>Uptime: {uptime_str} | Requests: {request_count}
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="card">
                    <div class="stat">
                        <div class="stat-value">{stats['total_requests']}</div>
                        <div class="stat-label">Requests (24h)</div>
                    </div>
                </div>
                <div class="card">
                    <div class="stat">
                        <div class="stat-value">{stats['total_tokens']:,}</div>
                        <div class="stat-label">Tokens (24h)</div>
                    </div>
                </div>
                <div class="card">
                    <div class="stat">
                        <div class="stat-value">${stats['total_cost']:.2f}</div>
                        <div class="stat-label">Cost (24h)</div>
                    </div>
                </div>
                <div class="card">
                    <div class="stat">
                        <div class="stat-value">{int(stats['avg_response_ms'])}ms</div>
                        <div class="stat-label">Avg Response</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Rate Limit Status</h2>
                {limits_html}
            </div>
            
            <div class="card">
                <h2>Usage by Provider (24h)</h2>
                <table>
                    <tr>
                        <th>Provider</th>
                        <th>Requests</th>
                        <th>Tokens</th>
                        <th>Cost</th>
                    </tr>
                    {provider_rows}
                </table>
            </div>
            
            <div class="card">
                <h2>API Endpoint</h2>
                <div class="endpoint">
                    POST http://localhost:8080/v1/chat/completions<br><br>
                    {{<br>
                    &nbsp;&nbsp;"model": "claude-opus-4-5",<br>
                    &nbsp;&nbsp;"messages": [...],<br>
                    &nbsp;&nbsp;"complexity": "simple" // or "complex"<br>
                    }}
                </div>
            </div>
        </body>
        </html>
        """
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def send_stats(self):
        """Send JSON stats"""
        stats = get_db_stats(24)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(stats).encode('utf-8'))
    
    def send_limits(self):
        """Send current rate limit status"""
        limits = {}
        for provider_key, provider in PROVIDERS.items():
            if not provider["api_key"]:
                continue
            current = get_provider_usage(provider_key)
            model = list(provider["models"].keys())[0]
            limit = provider["models"][model]["limit"]
            limits[provider_key] = {
                "current": current,
                "limit": limit,
                "available": limit - current,
                "percent_used": (current / limit) * 100
            }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(limits).encode('utf-8'))

def run_server(port: int = 8080):
    """Run the TokenWatch server"""
    init_db()
    
    server = HTTPServer(('0.0.0.0', port), TokenWatchHandler)
    print(f"⛽ TokenWatch running on http://localhost:{port}")
    print(f"📊 Dashboard: http://localhost:{port}/")
    print(f"🔌 API: http://localhost:{port}/v1/chat/completions")
    print(f"💾 Database: {DB_PATH}")
    print("\nPress Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⛔ Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    port = int(os.getenv("TOKENWATCH_PORT", 8080))
    run_server(port)
