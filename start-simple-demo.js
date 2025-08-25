#!/usr/bin/env node

const http = require('http');
const fs = require('fs');
const path = require('path');

const demoHtml = fs.readFileSync(path.join(__dirname, 'demo-app', 'index.html'), 'utf8');

const server = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(demoHtml);
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(3002, () => {
  console.log('✅ Demo app running at http://localhost:3002');
});

// Keep running until interrupted
process.on('SIGINT', () => {
  console.log('\n👋 Shutting down demo server');
  server.close();
  process.exit(0);
});