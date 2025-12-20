#!/bin/bash

# Start development server and display network information
# This script makes it easy to access the frontend from other devices

echo "=========================================="
echo "Starting Agri Robo Frontend Development Server"
echo "=========================================="
echo ""

# Get the Raspberry Pi's IP address
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# If hostname -I doesn't work, try alternative methods
if [ -z "$IP_ADDRESS" ]; then
    IP_ADDRESS=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}')
fi

if [ -z "$IP_ADDRESS" ]; then
    IP_ADDRESS=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
fi

echo "📍 Network Information:"
if [ -n "$IP_ADDRESS" ]; then
    echo "   Local IP: http://$IP_ADDRESS:3000"
    echo "   Localhost: http://localhost:3000"
else
    echo "   Could not detect IP address"
    echo "   Try: http://localhost:3000"
fi
echo ""
echo "💡 Access the website from any device on your network using:"
echo "   http://$IP_ADDRESS:3000"
echo ""
echo "=========================================="
echo "Starting Vite development server..."
echo "=========================================="
echo ""

# Start the dev server
npm run dev
