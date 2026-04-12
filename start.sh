#!/bin/bash

# PiAnnotate - Start Script

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/web"
ENV_NAME="piannotate"

echo "🎹 PiAnnotate Starting..."

# Activate conda environment
echo "🐍 Activating conda environment '$ENV_NAME'..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment '$ENV_NAME'"
    echo "   Please run 'bash setup.sh' first."
    exit 1
fi
echo "✅ Conda environment activated"
echo "   Project path: $SCRIPT_DIR"
echo "   Web path: $WEB_DIR"

# Kill existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "python server.py" 2>/dev/null
pkill -f "vite" 2>/dev/null
pkill -f "http.server 8080" 2>/dev/null

# Kill processes using ports 8080 and 3000 (using /proc if fuser not available)
kill_port() {
    local port=$1
    local hex_port=$(printf '%04X' $port)
    
    # Try fuser first
    if command -v fuser &> /dev/null; then
        fuser -k $port/tcp 2>/dev/null
        return
    fi
    
    # Fallback: use /proc
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
        if ls -la /proc/$pid/fd 2>/dev/null | grep -q "socket:"; then
            if cat /proc/net/tcp 2>/dev/null | grep -q ":$hex_port.*0A"; then
                if ls -la /proc/$pid/fd 2>/dev/null | grep -qE "socket:\[$(cat /proc/net/tcp | grep ":$hex_port" | awk '{print $10}' | head -1)\]"; then
                    kill -9 $pid 2>/dev/null
                fi
            fi
        fi
    done 2>/dev/null
}

kill_port 8080
kill_port 3000
sleep 1

# Start Flask backend (background)
echo "📦 Starting Flask backend (port 8080)..."
cd "$WEB_DIR"
python server.py > /tmp/flask.log 2>&1 &
FLASK_PID=$!

# Wait for Flask to start (check up to 10 seconds)
echo "   Waiting for Flask to start..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:8080 > /dev/null 2>&1; then
        echo "✅ Flask backend running (PID: $FLASK_PID)"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Flask backend failed to start. Check log:"
        cat /tmp/flask.log
        exit 1
    fi
done

# Start React frontend dev server
echo "⚛️  Starting React frontend (port 3000)..."
cd "$WEB_DIR"
npm run dev &
VITE_PID=$!

sleep 2

echo ""
echo "=========================================="
echo "🎉 All servers started!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8080"
echo ""
echo "   Press Ctrl+C to stop"
echo "=========================================="

# Cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $FLASK_PID 2>/dev/null
    kill $VITE_PID 2>/dev/null
    pkill -f "python server.py" 2>/dev/null
    pkill -f "vite" 2>/dev/null
    echo "👋 Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
